import os
import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import BertModel, BertTokenizerFast
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import logging
import math
import yaml
import time
import shutil
import ijson
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Pool, cpu_count
from utils import TAG_TO_IDX, IDX_TO_TAG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingPunctuationDataset(IterableDataset):
    """用于加载和预处理标点符号数据集的类"""
    def __init__(self, data_file, tokenizer, max_seq_length, cache_dir="cache/preprocessed",
                 num_workers=None, use_multiprocessing=True, buffer_size=1000, 
                 calculate_stats=True, shuffle=True):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)
        self.buffer_size = buffer_size
        self.calculate_stats = calculate_stats
        self.shuffle = shuffle

        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化标签分布计数器
        self.label_distribution = {tag: 0 for tag in TAG_TO_IDX.keys()}
        
        # 如果需要计算统计信息，预先扫描一次文件
        if self.calculate_stats:
            self._calculate_label_distribution()
            self._display_label_distribution()

    def __iter__(self):
        """迭代器方法，流式读取和处理数据"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # 打开文件并使用ijson解析
            with open(self.data_file, 'r', encoding='utf-8') as f:
                items = ijson.items(f, 'item')
                
                # 缓冲区用于在需要时打乱数据
                buffer = []
                
                for i, item in enumerate(items):
                    # 只处理属于当前worker的数据项
                    if i % num_workers == worker_id:
                        processed_item = self._process_single_item(item)
                        if processed_item is not None:
                            buffer.append(processed_item)
                            
                            # 当缓冲区达到指定大小时，打乱并返回数据
                            if len(buffer) >= self.buffer_size:
                                if self.shuffle:
                                    np.random.shuffle(buffer)
                                for example in buffer:
                                    yield example
                                buffer = []
                
                # 处理缓冲区中剩余的数据
                if buffer:
                    if self.shuffle:
                        np.random.shuffle(buffer)
                    for example in buffer:
                        yield example
        else:
            # 单worker情况，处理所有数据
            with open(self.data_file, 'r', encoding='utf-8') as f:
                items = ijson.items(f, 'item')
                
                buffer = []
                
                for item in items:
                    processed_item = self._process_single_item(item)
                    if processed_item is not None:
                        buffer.append(processed_item)
                        
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                np.random.shuffle(buffer)
                            for example in buffer:
                                yield example
                            buffer = []
                
                # 处理缓冲区中剩余的数据
                if buffer:
                    if self.shuffle:
                        np.random.shuffle(buffer)
                    for example in buffer:
                        yield example

    def _process_single_item(self, item):
        """处理单个数据项"""
        # 确保数据项包含必要的字段
        if not all(k in item for k in ["text", "labels"]):
            logger.warning(f"数据项缺少必要字段 'text' 或 'labels'")
            return None

        text = item["text"]
        labels = item["labels"]

        # 确保文本和标签长度一致
        if len(text) != len(labels):
            logger.warning(f"文本长度({len(text)})与标签长度({len(labels)})不匹配")
            return None

        # 将文本转换为token id
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # 提取offset映射和token ids
        offsets = tokenized["offset_mapping"][0]
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # 根据offset映射对齐标签
        aligned_labels = torch.zeros(len(input_ids), dtype=torch.long)
        aligned_labels.fill_(TAG_TO_IDX['O'])  # 默认填充为O标签

        # 对每个token，找到其对应的原始标签
        for i, (start, end) in enumerate(offsets):
            if start.item() == 0 and end.item() == 0:  # [PAD], [CLS], [SEP]等特殊token
                continue

            # 对应原始文本的索引
            token_start = start.item()
            token_end = end.item() - 1  # 包含end-1

            if token_start <= len(labels) - 1:
                # 获取标签（处理可能的列表情况）
                label = labels[token_start]
                
                # 如果标签是列表，取第一个元素
                if isinstance(label, list):
                    if label:  # 非空列表
                        label = label[0]
                    else:  # 空列表
                        label = 'O'
                
                # 将标签转换为字符串（以防是数字或其他类型）
                if not isinstance(label, str):
                    label = str(label)
                
                # 使用字符串标签查找索引
                aligned_labels[i] = TAG_TO_IDX.get(label, TAG_TO_IDX['O'])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": aligned_labels
        }

    def _calculate_label_distribution(self):
        """计算标签分布统计"""
        logger.info(f"计算标签分布统计...")
        
        # 检查是否存在缓存的标签分布
        label_dist_path = self._generate_label_dist_cache_path()
        if os.path.exists(label_dist_path):
            logger.info(f"从缓存加载标签分布: {label_dist_path}")
            with open(label_dist_path, 'rb') as f:
                self.label_distribution = pickle.load(f)
            return
        
        # 流式读取文件并计算标签分布
        with open(self.data_file, 'r', encoding='utf-8') as f:
            items = ijson.items(f, 'item')
            
            for item in tqdm(items, desc="计算标签分布"):
                if not all(k in item for k in ["text", "labels"]):
                    continue
                    
                labels = item["labels"]
                for label in labels:
                    # 处理可能的列表情况
                    if isinstance(label, list):
                        if label:  # 非空列表
                            label = label[0]
                        else:  # 空列表
                            label = 'O'
                    
                    # 将标签转换为字符串
                    if not isinstance(label, str):
                        label = str(label)
                    
                    # 更新计数
                    if label in self.label_distribution:
                        self.label_distribution[label] += 1
                    else:
                        self.label_distribution['O'] += 1
        
        # 缓存标签分布
        with open(label_dist_path, 'wb') as f:
            pickle.dump(self.label_distribution, f)
        
        logger.info(f"标签分布统计已保存至: {label_dist_path}")

    def _generate_label_dist_cache_path(self):
        """生成标签分布缓存文件路径"""
        filename = os.path.basename(self.data_file)
        return os.path.join(self.cache_dir, f"{filename}_{self.max_seq_length}_label_dist.pkl")

    def _display_label_distribution(self):
        """显示标签分布统计"""
        logger.info("标签分布统计:")

        total_count = sum(self.label_distribution.values())
        if total_count == 0:
            logger.warning("标签分布计算结果为空")
            return

        for tag, count in sorted(self.label_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_count) * 100
            logger.info(f"{tag} ({TAG_TO_IDX.get(tag, 'Unknown')}): {count} ({percentage:.2f}%)")

        # 特别关注O标签的比例
        o_count = self.label_distribution.get('O', 0)
        o_percentage = (o_count / total_count) * 100
        logger.info(f"\nO标签(背景标签)比例: {o_percentage:.2f}%")

        # 计算标签分布统计量
        counts = np.array([v for v in self.label_distribution.values() if v > 0])
        if len(counts) > 0:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            cv = std_count / mean_count if mean_count > 0 else 0

            logger.info(f"平均标签计数: {mean_count:.2f}")
            logger.info(f"标签计数标准差: {std_count:.2f}")
            logger.info(f"变异系数: {cv:.2f} (越小表示分布越均衡)")

            # 最大最小比
            max_count = np.max(counts)
            min_count = np.min(counts)
            max_min_ratio = max_count / min_count if min_count > 0 else float('inf')

            logger.info(f"最大计数: {max_count}")
            logger.info(f"最小计数: {min_count}")
            logger.info(f"最大/最小比: {max_min_ratio:.2f}")

        # 检查标签分布异常
        if o_percentage == 100:
            logger.warning("警告: 所有标签都是O标签(索引0)，这表明数据集中没有有效的标点符号标签")
            logger.warning("检查数据预处理步骤，确保标签正确转换")
        elif o_percentage > 95:
            logger.warning(f"警告: O标签占比过高 ({o_percentage:.2f}%)，这表明数据集中标点符号标签很少")
            logger.warning("检查数据预处理步骤，确保标签正确转换")


# 模型定义
class PunctuationModel(nn.Module):
    """用于标点符号预测的模型"""
    def __init__(self, config):
        super(PunctuationModel, self).__init__()
        
        # BERT模型
        self.bert = BertModel.from_pretrained(config['model']['bert_model_path'])
        
        # BERT输出层的维度
        hidden_size = config['model']['bert']['hidden_size']
        
        # Dropout - BERT后
        self.bert_dropout = nn.Dropout(config['model']['dropout'])
        
        # BiLSTM层
        lstm_config = config['model']['lstm']
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            bidirectional=lstm_config['bidirectional'],
            batch_first=lstm_config['batch_first'],
            dropout=lstm_config['dropout'] if lstm_config['num_layers'] > 1 else 0
        )
        
        # 层归一化 - LSTM后
        lstm_output_size = lstm_config['hidden_size'] * 2 if lstm_config['bidirectional'] else lstm_config['hidden_size']
        self.lstm_layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Dropout - LSTM后
        self.lstm_dropout = nn.Dropout(config['model']['dropout'])
        
        # 多头自注意力
        attention_config = config['model']['attention']
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_config['embed_dim'],
            num_heads=attention_config['num_heads'],
            dropout=attention_config['dropout'],
            batch_first=attention_config['batch_first']
        )
        
        # 层归一化 - 注意力后
        self.attention_layer_norm = nn.LayerNorm(attention_config['embed_dim'])
        
        # Dropout - 注意力后
        self.attention_dropout = nn.Dropout(config['model']['dropout'])
        
        # 全连接层
        self.classifier = nn.Linear(attention_config['embed_dim'], config['model']['num_tags'])
        
        # 冻结BERT底层
        freeze_layers = config['training']['freeze_bert_layers']
        if freeze_layers > 0:
            logger.info(f"冻结BERT底部的{freeze_layers}层")
            modules = [self.bert.embeddings, *self.bert.encoder.layer[:freeze_layers]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """前向传播"""
        # BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.bert_dropout(sequence_output)
        
        # BiLSTM前向传播
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.lstm_layer_norm(lstm_output)
        lstm_output = self.lstm_dropout(lstm_output)
        
        # 多头自注意力
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        attn_output, _ = self.attention(
            lstm_output, 
            lstm_output, 
            lstm_output, 
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # 残差连接 + 层归一化
        combined_output = lstm_output + attn_output
        combined_output = self.attention_layer_norm(combined_output)
        combined_output = self.attention_dropout(combined_output)
        
        # 分类层
        logits = self.classifier(combined_output)
        
        # 始终将logits作为第一个返回值，以便HybridLoss可以正确处理
        outputs = (logits,)

        # 如果提供了标签，计算损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 只计算非padding位置的损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            # 将loss作为第一个元素，logits作为第二个元素
            outputs = (loss, logits)

        return outputs


class HybridLoss(nn.Module):
    """结合Focal Loss和加权交叉熵的联合损失函数"""
    def __init__(self, alpha=0.5, gamma=2.0, class_weights=None, device=None):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.device = device

        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, logits, targets):
        """计算联合损失"""
        # 添加维度检查和错误处理
        if logits.dim() == 0 or logits.size(0) == 0:
            logger.error(f"收到空的logits张量: shape={logits.shape}")
            # 返回零损失，避免训练中断
            return torch.tensor(0.0, device=self.device if self.device else logits.device)
            
        # 确保logits有正确的维度
        if logits.dim() < 2:
            logger.error(f"logits维度不足: shape={logits.shape}")
            # 尝试添加必要的维度
            logits = logits.unsqueeze(0)
            if logits.dim() < 2:
                logits = logits.unsqueeze(-1)
                
        # 计算加权交叉熵损失
        try:
            ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # 计算Focal Loss部分
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
            # 组合两种损失
            loss = self.alpha * focal_loss + (1 - self.alpha) * ce_loss
            
            return loss.mean()
        except Exception as e:
            logger.error(f"计算损失时出错: {e}")
            logger.error(f"logits shape: {logits.shape}, targets shape: {targets.shape}")
            # 返回零损失，避免训练中断
            return torch.tensor(0.0, device=self.device if self.device else logits.device)


def calculate_class_weights(label_distribution, num_classes, method='log_scale', beta=0.9, max_weight=10.0):
    """权重计算"""
    # 确保所有类别都有计数
    counts = np.zeros(num_classes)
    for tag, idx in TAG_TO_IDX.items():
        if idx < num_classes:
            counts[idx] = label_distribution.get(tag, 0)

    # 避免零除错误
    counts = np.where(counts == 0, 1, counts)

    if method == 'log_scale':
        # 对数缩放
        weights = 1.0 / np.log(1.1 + counts)
    elif method == 'effective_samples':
        # 有效样本数方法
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.where(effective_num == 0, 1e-8, effective_num)
    else:
        # 倒数加权
        weights = 1.0 / counts

    # 归一化权重
    weights = weights / np.sum(weights) * num_classes

    # 限制最大权重
    weights = np.minimum(weights, max_weight)

    # 转换为PyTorch张量
    return torch.FloatTensor(weights)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, last_epoch=-1):
    """创建带预热的余弦学习率调度器"""
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio

        return decayed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def evaluate(model, data_loader, device, num_tags):
    """评估模型性能"""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # 检查outputs的结构并正确提取logits
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                # 如果outputs是包含loss和logits的元组
                logits = outputs[1]
            else:
                # 如果outputs只包含logits
                logits = outputs[0]
                
            predictions = torch.argmax(logits, dim=-1)

            # 收集非padding位置的预测和标签
            active_positions = batch["attention_mask"].view(-1) == 1
            active_preds = predictions.view(-1)[active_positions]
            active_labels = batch["labels"].view(-1)[active_positions]

            all_predictions.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())

    # 计算各项指标
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)

    # 生成详细报告
    report = classification_report(
        all_labels, all_predictions,
        labels=list(range(num_tags)),
        target_names=list(IDX_TO_TAG.values()),
        zero_division=0
    )

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'report': report
    }

    return metrics


def collate_fn(batch):
    """数据加载器的整理函数"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def save_checkpoint(model, optimizer, scheduler, config, epoch, metrics, best_f1, checkpoint_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
        'metrics': metrics,
        'best_f1': best_f1
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"模型检查点已保存至 {checkpoint_path}")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['best_f1'], checkpoint.get('metrics', {})


def train(config):
    # 设置随机种子
    torch.manual_seed(config['training']['seed'])
    torch.cuda.manual_seed_all(config['training']['seed'])

    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建必要的目录
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # 初始化TensorBoard
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_writer = SummaryWriter(f"runs/{run_name}")
    logger.info(f"TensorBoard日志将保存在 runs/{run_name}")

    # 初始化分词器
    tokenizer = BertTokenizerFast.from_pretrained(config['model']['bert_model_path'])

    # 根据配置选择数据集类型
    use_streaming = config['training'].get('use_streaming', False)
    
    # 加载数据集
    logger.info("加载训练集...")
    if use_streaming:
        train_dataset = StreamingPunctuationDataset(
            config['data']['train_file'],
            tokenizer,
            config['model']['max_seq_length'],
            cache_dir=config['data']['cache_dir'],
            use_multiprocessing=True,
            buffer_size=config['training'].get('buffer_size', 1000),
            calculate_stats=True,
            shuffle=True
        )

    logger.info("加载验证集...")
    if use_streaming:
        dev_dataset = StreamingPunctuationDataset(
            config['data']['dev_file'],
            tokenizer,
            config['model']['max_seq_length'],
            cache_dir=config['data']['cache_dir'],
            use_multiprocessing=True,
            buffer_size=config['training'].get('buffer_size', 1000),
            calculate_stats=True,
            shuffle=False
        )

    # 计算类别权重
    class_weights = calculate_class_weights(
        train_dataset.label_distribution,
        config['model']['num_tags'],
        method=config['training']['class_weight_method'],
        max_weight=config['training']['max_weight']
    )

    # 保存类别权重
    weight_cache_path = os.path.join(config['data']['cache_dir'], 'class_weights.pkl')
    with open(weight_cache_path, 'wb') as f:
        pickle.dump(class_weights, f)
    logger.info(f"类别权重已保存至 {weight_cache_path}")

    # 创建数据加载器 - 根据数据集类型调整
    if use_streaming:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config['training'].get('dataloader_workers', 2),
            pin_memory=True,
            prefetch_factor=config['training'].get('prefetch_factor', 2)  # 预取批次数
        )

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config['training'].get('dataloader_workers', 2),
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2
        )

        dev_loader = DataLoader(
            dev_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )

    # 初始化模型
    model = PunctuationModel(config)
    model.to(device)

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )

    # 设置学习率调度器
    if use_streaming:
        # 对于流式数据集，需要估计每个epoch的步数
        steps_per_epoch = config['training'].get('steps_per_epoch', 1000)
        num_training_steps = steps_per_epoch * config['training']['epochs']
    else:
        # 对于常规数据集，我直接计算步数
        num_training_steps = len(train_loader) * config['training']['epochs']
        
    num_warmup_steps = int(num_training_steps * config['training']['lr_scheduler']['warmup_ratio'])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=config['training']['lr_scheduler']['min_lr_ratio']
    )

    # 将类别权重移至设备
    class_weights = class_weights.to(device)

    # 初始化损失函数
    loss_fn = HybridLoss(
        alpha=config['training']['hybrid_loss_alpha'],
        gamma=config['training']['focal_loss_gamma'],
        class_weights=class_weights,
        device=device
    )

    # 混合精度训练设置
    scaler = torch.amp.GradScaler() if config['training']['fp16'] and torch.cuda.is_available() else None

    # 保存初始模型
    initial_checkpoint_path = os.path.join(config['training']['save_dir'], 'initial_model.pth')
    save_checkpoint(model, optimizer, scheduler, config, 0, {}, 0.0, initial_checkpoint_path)
    
    # 跟踪最佳模型
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 全局步数计数器（用于TensorBoard）
    global_step = 0
    
    # 训练循环
    for epoch in range(1, config['training']['epochs'] + 1):
        logger.info(f"开始第 {epoch}/{config['training']['epochs']} 轮训练")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_steps = 0

        # 根据数据集类型选择不同的训练流程
        if use_streaming:
            # 流式数据集训练流程
            # 使用tqdm创建进度条
            progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
            
            # 限制每个epoch的步数，避免无限循环
            max_steps_per_epoch = config['training'].get('max_steps_per_epoch', steps_per_epoch)
            
            for batch_idx, batch in enumerate(progress_bar):
                # 检查是否达到最大步数
                if batch_idx >= max_steps_per_epoch:
                    break
                    
                batch = {k: v.to(device) for k, v in batch.items()}

                # 清零梯度
                optimizer.zero_grad()

                # 混合精度训练
                if scaler:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )
                        # 检查outputs的结构并正确提取logits
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            # 如果outputs是包含loss和logits的元组
                            loss, logits = outputs[0], outputs[1]
                        else:
                            # 如果outputs只包含logits
                            logits = outputs[0]
                            loss = loss_fn(logits, batch["labels"])

                    # 反向传播
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    if config['training']['gradient_clip_val'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])

                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准训练
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    # 检查outputs的结构并正确提取logits
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        # 如果outputs是包含loss和logits的元组
                        loss, logits = outputs[0], outputs[1]
                    else:
                        # 如果outputs只包含logits
                        logits = outputs[0]
                        loss = loss_fn(logits, batch["labels"])

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    if config['training']['gradient_clip_val'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])

                    # 更新参数
                    optimizer.step()

                # 更新学习率
                scheduler.step()
                
                # 记录到TensorBoard
                current_lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar('training/learning_rate', current_lr, global_step)
                tb_writer.add_scalar('training/batch_loss', loss.item(), global_step)
                
                # 更新全局步数
                global_step += 1

                # 更新统计信息
                train_loss += loss.item()
                train_steps += 1

                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})
                
                # 定期释放缓存，减少内存占用
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # 常规数据集训练流程
            progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")

            optimizer_stepped = False
            first_batch_scheduler_stepped = False
            
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}

                # 清零梯度
                optimizer.zero_grad()

                # 混合精度训练
                if scaler:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )
                        # 检查outputs的结构并正确提取logits
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            # 如果outputs是包含loss和logits的元组
                            loss, logits = outputs[0], outputs[1]
                        else:
                            # 如果outputs只包含logits
                            logits = outputs[0]
                            loss = loss_fn(logits, batch["labels"])

                    # 反向传播
                    scaler.scale(loss).backward()

                    # 梯度裁剪
                    if config['training']['gradient_clip_val'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])

                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_stepped = True
                else:
                    # 标准训练
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    # 检查outputs的结构并正确提取logits
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        # 如果outputs是包含loss和logits的元组
                        loss, logits = outputs[0], outputs[1]
                    else:
                        # 如果outputs只包含logits
                        logits = outputs[0]
                        loss = loss_fn(logits, batch["labels"])

                    # 反向传播
                    loss.backward()

                    # 梯度裁剪
                    if config['training']['gradient_clip_val'] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])

                    # 更新参数
                    optimizer.step()
                    optimizer_stepped = True
                
                # 记录到TensorBoard
                current_lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar('training/learning_rate', current_lr, global_step)
                tb_writer.add_scalar('training/batch_loss', loss.item(), global_step)
                
                # 更新全局步数
                global_step += 1

                # 更新统计信息
                train_loss += loss.item()
                train_steps += 1

                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if train_steps == 1 and epoch == 1 and optimizer_stepped and not first_batch_scheduler_stepped:
                    scheduler.step()
                    first_batch_scheduler_stepped = True

            # 在每个epoch结束时更新学习率
            if optimizer_stepped:
                if not (epoch == 1 and first_batch_scheduler_stepped):
                    scheduler.step()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        logger.info(f"Epoch {epoch} - 平均训练损失: {avg_train_loss:.4f}")
        
        # 记录平均训练损失到TensorBoard
        tb_writer.add_scalar('training/avg_train_loss', avg_train_loss, epoch)

        # 评估阶段
        logger.info("评估模型...")
        
        # 根据数据集类型选择不同的评估函数
        if use_streaming:
            # 对于流式数据集，需要限制评估的批次数
            max_eval_steps = config['training'].get('max_eval_steps', 100)
            metrics = evaluate_streaming(model, dev_loader, device, config['model']['num_tags'], max_steps=max_eval_steps)
        else:
            # 对于常规数据集，使用标准评估函数
            metrics = evaluate(model, dev_loader, device, config['model']['num_tags'])

        # 打印评估结果
        logger.info(
            f"Epoch {epoch} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 准确率: {metrics['accuracy']:.4f}")
        logger.info(f"详细评估报告:\n{metrics['report']}")
        
        # 记录评估指标到TensorBoard
        tb_writer.add_scalar('evaluation/precision', metrics['precision'], epoch)
        tb_writer.add_scalar('evaluation/recall', metrics['recall'], epoch)
        tb_writer.add_scalar('evaluation/f1', metrics['f1'], epoch)
        tb_writer.add_scalar('evaluation/accuracy', metrics['accuracy'], epoch)

        # 保存当前模型（覆盖之前的模型）
        current_checkpoint_path = os.path.join(config['training']['save_dir'], 'latest_model.pth')
        save_checkpoint(model, optimizer, scheduler, config, epoch, metrics, best_f1, current_checkpoint_path)
        logger.info(f"已保存第 {epoch} 轮模型，覆盖之前的模型")

        # 检查是否为最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_epoch = epoch
            patience_counter = 0

            # 保存最佳模型
            best_checkpoint_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
            shutil.copy(current_checkpoint_path, best_checkpoint_path)
            logger.info(f"找到新的最佳模型，F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"没有改进，当前耐心计数: {patience_counter}/{config['training']['patience']}")

        # 早停机制
        if patience_counter >= config['training']['patience']:
            logger.info(f"达到早停条件，{config['training']['patience']}轮没有改进，停止训练")
            break

    # 关闭TensorBoard写入器
    tb_writer.close()
    
    # 训练完成
    logger.info(f"训练完成，最佳模型在第{best_epoch}轮，F1: {best_f1:.4f}")

    return best_f1, best_epoch


def evaluate_streaming(model, data_loader, device, num_tags, max_steps=100):
    """评估流式数据集上的模型性能"""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="评估", total=max_steps)):
            # 限制评估步数
            if batch_idx >= max_steps:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # 检查outputs的结构并正确提取logits
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                # 如果outputs是包含loss和logits的元组
                logits = outputs[1]
            else:
                # 如果outputs只包含logits
                logits = outputs[0]
                
            predictions = torch.argmax(logits, dim=-1)

            # 收集非padding位置的预测和标签
            active_positions = batch["attention_mask"].view(-1) == 1
            active_preds = predictions.view(-1)[active_positions]
            active_labels = batch["labels"].view(-1)[active_positions]

            all_predictions.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())
            
            # 定期释放缓存，减少内存占用
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 计算各项指标
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_predictions)

    # 生成详细报告
    report = classification_report(
        all_labels, all_predictions,
        labels=list(range(num_tags)),
        target_names=list(IDX_TO_TAG.values()),
        zero_division=0
    )

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'report': report
    }

    return metrics


def main():
    """主函数"""
    # 加载配置
    config_path = "configs/config.yml"
    config = load_config(config_path)

    # 添加依赖检查
    try:
        import ijson
        logger.info("ijson库已安装，可以使用流式数据处理")
    except ImportError:
        logger.warning("未安装ijson库，无法使用流式数据处理。将使用常规数据处理。")
        logger.warning("请使用 pip install ijson 安装")
        config['training']['use_streaming'] = False

    # 检查是否使用流式数据处理
    if config['training'].get('use_streaming', False):
        logger.info("使用流式数据处理模式")
    else:
        logger.info("使用常规数据处理模式")

    # 开始训练
    start_time = time.time()
    best_f1, best_epoch = train(config)
    end_time = time.time()

    # 打印训练结果
    logger.info(f"训练耗时: {(end_time - start_time) / 60:.2f} 分钟")
    logger.info(f"最佳F1分数: {best_f1:.4f} (Epoch {best_epoch})")


if __name__ == "__main__":
    main()
