import torch
import json
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from model import Model
from utils import TAG_TO_IDX
from transformers import BertTokenizerFast
import yaml
import logging

# 加载配置文件
with open('configs/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# 加载BERT分词器
tokenizer = BertTokenizerFast.from_pretrained(config['model']['bert_model_path'])

# 创建模型
model = Model(config)
model.init_crf_transitions(config_dict=config)

# 加载模型权重
try:
    model.load_state_dict(torch.load(config['model']['save_path']))
    logging.info(f"模型权重已从 {config['model']['save_path']} 加载")
except Exception as e:
    logging.error(f"加载模型权重失败: {e}")
    raise

# 检查可用的GPU数量
num_gpus = torch.cuda.device_count()
logging.info(f'可用的GPU数量: {num_gpus}')

# 使用DataParallel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if num_gpus > 1:
    model = torch.nn.DataParallel(model)
    logging.info('使用DataParallel进行多卡评估')

# 将模型移动到设备
model.to(device)
logging.info(f'模型已移动到设备: {device}')

# 获取CRF模块引用
crf_module = model.module.crf if isinstance(model, torch.nn.DataParallel) else model.crf

model.eval()

# 创建反向映射
IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}

# 忽略的标签索引
IGNORE_INDEX = -100


def evaluate(data_file):
    """评估模型在给定数据集上的性能
    
    Args:
        data_file: 数据文件路径
        
    Returns:
        tuple: (f1分数, 准确率)
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"加载数据文件失败: {str(e)}")
        return 0.0, 0.0

    if not data:
        logging.warning("数据文件为空")
        return 0.0, 0.0

    all_predictions = []
    all_labels = []
    
    # 验证标签映射
    for tag in TAG_TO_IDX.values():
        if not isinstance(tag, int) or tag < 0:
            logging.error(f"无效的标签索引: {tag}")
            return 0.0, 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for item in tqdm(data, desc="评估样本", leave=True):
        try:
            if 'text' not in item or 'labels' not in item:
                logging.warning(f"数据项缺少必要的字段: {item}")
                continue
                
            text = item['text']
            true_labels = item['labels'].split()

            # 预处理文本
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=config['model']['max_seq_length'],
                return_tensors='pt',
                return_offsets_mapping=True
            )
            
            offset_mapping = encoding.pop('offset_mapping').squeeze(0)
            input_ids = encoding['input_ids'].squeeze(0).to(device)
            attention_mask = encoding['attention_mask'].squeeze(0).to(device)

            # 验证输入数据
            if input_ids.size(0) > config['model']['max_seq_length']:
                logging.warning(f"输入序列长度超过最大长度: {input_ids.size(0)}")
                continue

            # 创建标签和掩码
            label_ids = []
            label_index = 0
            
            for i, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    label_ids.append(IGNORE_INDEX)
                elif label_index < len(true_labels):
                    label = true_labels[label_index]
                    if label not in TAG_TO_IDX:
                        logging.warning(f"未知的标签: {label}")
                        label_id = len(TAG_TO_IDX)  # 使用默认标签
                    else:
                        label_id = TAG_TO_IDX[label]
                    label_ids.append(label_id)
                    label_index += 1
                else:
                    label_ids.append(IGNORE_INDEX)
                    
            label_ids = torch.tensor(label_ids, dtype=torch.long).to(device)
            mask = (label_ids != IGNORE_INDEX).bool().to(device)

            with torch.no_grad():
                try:
                    outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), tags=None, mask=mask.unsqueeze(0))
                    
                    # 验证模型输出
                    if not isinstance(outputs, dict) or 'logits' not in outputs:
                        logging.warning("模型输出格式不正确")
                        continue
                        
                    logits = outputs['logits']
                    
                    # 检查NaN值
                    if torch.isnan(logits).any():
                        logging.warning("检测到NaN值在logits中")
                        continue
                        
                    probs = torch.sigmoid(logits)
                    predictions = (probs > 0.5).long().squeeze(0)
                except Exception as e:
                    logging.error(f"模型推理失败: {str(e)}")
                    continue
            
            # 将预测结果和真实标签转换为列表
            pred_tags = []
            true_tags = []
            
            for i, (pred, label) in enumerate(zip(predictions, label_ids.cpu().tolist())):
                if label != IGNORE_INDEX and mask[i]:
                    if pred.item() not in IDX_TO_TAG:
                        logging.warning(f"未知的预测标签索引: {pred.item()}")
                        continue
                    pred_tags.append(IDX_TO_TAG[pred.item()])
                    true_tags.append(IDX_TO_TAG[label])
        
            all_predictions.extend(pred_tags)
            all_labels.extend(true_tags)
            
        except Exception as e:
            logging.error(f"处理样本时发生错误: {str(e)}")
            continue

    if not all_predictions or not all_labels:
        logging.warning("没有有效的预测结果")
        return 0.0, 0.0

    try:
        # 计算评估指标
        f1 = f1_score(all_labels, all_predictions, average='macro')
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 记录详细的评估结果
        logging.info(f"评估完成:")
        logging.info(f"样本总数: {len(data)}")
        logging.info(f"有效预测数: {len(all_predictions)}")
        logging.info(f"F1分数: {f1:.4f}")
        logging.info(f"准确率: {accuracy:.4f}")
        
        return f1, accuracy
        
    except Exception as e:
        logging.error(f"计算评估指标时发生错误: {str(e)}")
        return 0.0, 0.0

if __name__ == '__main__':
    evaluate(config['data']['test_file'])  # 使用测试集进行评估
