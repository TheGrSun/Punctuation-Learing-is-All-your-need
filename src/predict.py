import os
import torch
import yaml
import logging
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast
import re
from utils import TAG_TO_IDX, IDX_TO_TAG, PUNCTUATION_MAP

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 反向映射，从标签到实际标点
TAG_TO_PUNCTUATION = {tag: punct for punct, tag in PUNCTUATION_MAP.items()}

class PunctuationRestorer:
    def __init__(self, model_path, config_path):
        # 加载配置
        self.config = self._load_config(config_path)

        # 验证标签映射
        self._verify_punctuation_mapping()

        # 初始化分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config['model']['bert_model_path'])

        # 加载模型
        self.model, self.device = self._load_model(model_path)

        # 设置最大序列长度
        self.max_seq_length = self.config['model']['max_seq_length']

        # 设置滑动窗口大小和步长
        self.window_size = self.config['data'].get('window_size', 50)
        self.stride = self.window_size // 2  # 窗口重叠一半

        # 检查是否使用GPU
        self.use_gpu = torch.cuda.is_available()
        logger.info(f"使用设备: {self.device}")

        logger.info("标点恢复器初始化完成")

    def _verify_punctuation_mapping(self):
        """验证标签到标点符号的映射是否完整"""
        logger.info("验证标签到标点符号的映射...")

        # 确保所有非O标签都有对应的标点符号映射
        missing_tags = []
        for tag in TAG_TO_IDX.keys():
            if tag != 'O' and tag not in TAG_TO_PUNCTUATION:
                missing_tags.append(tag)

        if missing_tags:
            logger.warning(f"以下标签在TAG_TO_PUNCTUATION中缺少映射: {missing_tags}")
            logger.warning("将为缺失的标签添加默认映射...")

            # 为缺失的标签添加默认映射
            for tag in missing_tags:
                if tag.startswith('B-DQUOTE'):
                    TAG_TO_PUNCTUATION[tag] = '\u201c'  # 左双引号
                elif tag.startswith('E-DQUOTE'):
                    TAG_TO_PUNCTUATION[tag] = '\u201d'  # 右双引号
                elif tag.startswith('B-SQUOTE'):
                    TAG_TO_PUNCTUATION[tag] = '\u2018'  # 左单引号
                elif tag.startswith('E-SQUOTE'):
                    TAG_TO_PUNCTUATION[tag] = '\u2019'  # 右单引号
                elif tag.startswith('B-BRACKET'):
                    TAG_TO_PUNCTUATION[tag] = '（'  # 左括号
                elif tag.startswith('E-BRACKET'):
                    TAG_TO_PUNCTUATION[tag] = '）'  # 右括号
                elif tag.startswith('B-BOOK'):
                    TAG_TO_PUNCTUATION[tag] = '《'
                elif tag.startswith('E-BOOK'):
                    TAG_TO_PUNCTUATION[tag] = '》'
                elif tag.startswith('S-'):
                    # 根据标签名称设置一个合理的默认标点
                    if 'COMMA' in tag.upper():
                        TAG_TO_PUNCTUATION[tag] = '，'
                    elif 'PERIOD' in tag.upper():
                        TAG_TO_PUNCTUATION[tag] = '。'
                    elif 'QUESTION' in tag.upper():
                        TAG_TO_PUNCTUATION[tag] = '？'
                    elif 'EXCLAM' in tag.upper():
                        TAG_TO_PUNCTUATION[tag] = '！'
                    else:
                        TAG_TO_PUNCTUATION[tag] = '，'  # 默认使用逗号
                else:
                    # 为其他未知标签设置默认标点
                    TAG_TO_PUNCTUATION[tag] = '，'

            logger.info("已为所有缺失的标签添加默认映射")
        else:
            logger.info("标签到标点符号的映射验证完成，所有标签都有对应的映射")

    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _load_model(self, model_path):
        """加载模型"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # 尝试加载checkpoint
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                logger.info("使用 weights_only=True 成功加载模型")
            except Exception as e:
                logger.warning(f"使用 weights_only=True 加载失败: {e}")
                logger.warning("尝试使用 weights_only=False 加载模型")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                logger.info("使用 weights_only=False 成功加载模型")

            # 从训练脚本中导入模型类
            try:
                from src.train import PunctuationModel
                model = PunctuationModel(self.config)
            except ImportError:
                logger.warning("从 src.train 导入 PunctuationModel 失败，尝试从 train 导入")
                from train import PunctuationModel
                model = PunctuationModel(self.config)

            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info(f"模型已从 {model_path} 加载")

            # 将模型移动到设备并设置为评估模式
            model.to(device)
            model.eval()

            return model, device
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.error("请确保模型文件存在且格式正确")
            raise

    def _preprocess_text(self, text):
        """预处理文本，移除已有标点，合并空格"""
        # 检查文本是否为空
        if not text or len(text.strip()) == 0:
            logger.warning("输入文本为空，无法处理")
            return ""

        # 添加英文标点到中文标点的转换
        english_to_chinese = {
            ':': '：',
            ';': '；',
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            '(': '（',
            ')': '）',
            '[': '【',
            ']': '】',
            '{': '｛',
            '}': '｝',
            '...': '……'
        }
        
        # 先规范化英文标点（保留特殊格式）
        for eng, chn in english_to_chinese.items():
            # 跳过时间格式中的冒号
            if eng == ':' and re.search(r'\d+:\d+', text):
                continue
            text = text.replace(eng, chn)
        
        # 然后移除所有标点进行预处理
        for punct in PUNCTUATION_MAP.keys():
            text = text.replace(punct, "")

        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)

        processed_text = text.strip()

        # 检查处理后的文本是否为空
        if not processed_text:
            logger.warning("处理后的文本为空，请检查输入")
            return ""

        return processed_text

    def _split_text_into_windows(self, text):
        """将文本分割成滑动窗口"""
        text_length = len(text)
        windows = []
        char_indices = []

        # 如果文本长度小于最大序列长度，直接处理整个文本
        if text_length <= self.max_seq_length:
            windows.append(text)
            char_indices.append(list(range(text_length)))
            return windows, char_indices

        # 滑动窗口处理
        start = 0
        while start < text_length:
            end = min(start + self.window_size, text_length)

            # 收集当前窗口的字符和对应的原始索引
            window_text = text[start:end]
            indices = list(range(start, end))

            windows.append(window_text)
            char_indices.append(indices)

            # 更新开始位置
            start += self.stride

            # 如果剩余文本不足一个完整窗口，且不是最后一段，则跳过
            if start < text_length and text_length - start < self.window_size // 4:
                break

        return windows, char_indices

    def _predict_window(self, text_window):
        """预测单个文本窗口的标点"""
        # 分词
        inputs = self.tokenizer(
            text_window,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # 将输入移到设备上
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offsets = inputs["offset_mapping"][0]

        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # 检查outputs的结构并正确提取logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # 对logits进行argmax得到预测的标签索引
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # 只保留实际文本对应的预测结果，忽略padding
        valid_predictions = []
        valid_positions = []

        for i, (start, end) in enumerate(offsets):
            # 跳过特殊token如[CLS], [SEP], [PAD]
            if start.item() == 0 and end.item() == 0:
                continue

            # 有效字符的位置
            char_pos = start.item()

            if char_pos < len(text_window):
                valid_positions.append(char_pos)
                valid_predictions.append(predictions[i])

        return valid_positions, valid_predictions

    def _merge_window_predictions(self, text, windows_char_indices, windows_predictions):
        """合并多个窗口的预测结果"""
        text_length = len(text)
        merged_predictions = [0] * text_length  # 默认为"O"标签(索引0)
        prediction_counts = [0] * text_length  # 记录每个位置被预测的次数

        # 合并多个窗口的预测
        for window_indices, (positions, predictions) in zip(windows_char_indices, windows_predictions):
            for pos, pred in zip(positions, predictions):
                if pos < len(window_indices):
                    orig_idx = window_indices[pos]
                    if orig_idx < text_length:
                        merged_predictions[orig_idx] += pred
                        prediction_counts[orig_idx] += 1

        # 计算每个位置的平均预测结果
        final_predictions = []
        for i in range(text_length):
            if prediction_counts[i] > 0:
                # 取众数而不是平均值
                final_predictions.append(merged_predictions[i] // prediction_counts[i])
            else:
                final_predictions.append(0)  # 默认为"O"标签

        return final_predictions

    def _insert_punctuation(self, text, predictions):
        """根据预测结果在文本中插入标点"""
        result = []
        
        # 创建标点对应关系字典
        paired_punctuation = {
            'B-DQUOTE_L': 'E-DQUOTE_R',  # 双引号
            'B-SQUOTE_L': 'E-SQUOTE_R',  # 单引号
            'B-BRACKET_L': 'E-BRACKET_R',  # 括号
            'B-BOOK_L': 'E-BOOK_R',  # 书名号
            'B-BRACE_L': 'E-BRACE_R',  # 大括号
            'B-MBRACKET_L': 'E-MBRACKET_R'  # 中括号
        }
        
        # 创建反向映射
        reverse_paired = {v: k for k, v in paired_punctuation.items()}
        
        # 存储所有开始标点的位置
        open_punct_positions = {}
        for k in paired_punctuation.keys():
            open_punct_positions[k] = []
        
        # 存储所有结束标点的位置
        close_punct_positions = {}
        for k in paired_punctuation.values():
            close_punct_positions[k] = []
            
        # 查找所有标点位置并修正标签
        for i, char in enumerate(text):
            if i < len(predictions) and predictions[i] != 0:  # 不是O标签
                tag = IDX_TO_TAG[predictions[i]]
                
                # 修正可能错误的标签（如右引号被错误预测为左引号）
                if i > 0 and tag.startswith('E-') and i+1 < len(text):
                    # 检查前后文本特征，判断是否应该是开始标点
                    prev_char = text[i-1]
                    next_char = text[i+1] if i+1 < len(text) else ""
                    
                    # 如果前面是句号、逗号等，后面是字母或汉字，可能是开始标点被误识别
                    if (prev_char in "。，！？；：" and next_char.isalnum()):
                        # 将结束标点改为对应的开始标点
                        if tag == 'E-DQUOTE_R':
                            tag = 'B-DQUOTE_L'
                            predictions[i] = TAG_TO_IDX[tag]
                        elif tag == 'E-SQUOTE_R':
                            tag = 'B-SQUOTE_L'
                            predictions[i] = TAG_TO_IDX[tag]
                
                # 同样检查开始标点是否应该是结束标点
                if i > 0 and tag.startswith('B-'):
                    prev_char = text[i-1]
                    # 如果前面不是句号、逗号等，可能是结束标点被误识别
                    if prev_char.isalnum() and (i+1 == len(text) or text[i+1] in "。，！？；："):
                        # 将开始标点改为对应的结束标点
                        if tag == 'B-DQUOTE_L':
                            tag = 'E-DQUOTE_R'
                            predictions[i] = TAG_TO_IDX[tag]
                        elif tag == 'B-SQUOTE_L':
                            tag = 'E-SQUOTE_R'
                            predictions[i] = TAG_TO_IDX[tag]
                
                # 记录标点位置
                if tag in paired_punctuation:
                    open_punct_positions[tag].append(i)
                elif tag in reverse_paired:
                    close_punct_positions[tag].append(i)
        
        # 增强的标点平衡处理
        for start_tag, end_tag in paired_punctuation.items():
            start_positions = open_punct_positions[start_tag]
            end_positions = close_punct_positions[end_tag]
            
            # 如果没有开始或结束标点，跳过处理
            if not start_positions and not end_positions:
                continue
                
            # 强制平衡：如果只有开始标点或只有结束标点，全部删除
            if not start_positions or not end_positions:
                # 删除所有不成对的标点
                for pos in start_positions + end_positions:
                    predictions[pos] = 0
                logger.debug(f"删除所有不成对的标点 {start_tag}/{end_tag}")
                continue
                
            # 计算有效配对
            valid_pairs = []
            remaining_starts = start_positions.copy()
            remaining_ends = end_positions.copy()
            
            # 优先匹配距离合理的配对
            for start_pos in sorted(start_positions):
                # 找到当前开始标点之后的第一个结束标点
                valid_ends = [end for end in remaining_ends if end > start_pos]
                
                if valid_ends:
                    # 选择最近的有效结束标点
                    end_pos = min(valid_ends)
                    
                    # 检查距离是否合理（根据不同标点类型可能有不同阈值）
                    max_distance = 500  # 最大允许距离
                    if start_tag.startswith('B-BOOK'):  # 书名号通常跨度较小
                        max_distance = 200
                    elif start_tag.startswith('B-DQUOTE') or start_tag.startswith('B-SQUOTE'):  # 引号可能跨度较大
                        max_distance = 400
                        
                    if end_pos - start_pos <= max_distance:
                        valid_pairs.append((start_pos, end_pos))
                        remaining_starts.remove(start_pos)
                        remaining_ends.remove(end_pos)
            
            # 处理剩余未配对的标点
            # 1. 删除多余的开始标点（从后向前）
            for start_pos in sorted(remaining_starts, reverse=True):
                predictions[start_pos] = 0
                logger.debug(f"删除未闭合的开始标点 {start_tag} 在位置 {start_pos}")
            
            # 2. 删除多余的结束标点（从后向前）
            for end_pos in sorted(remaining_ends, reverse=True):
                predictions[end_pos] = 0
                logger.debug(f"删除未闭合的结束标点 {end_tag} 在位置 {end_pos}")
        
        # 根据处理后的预测插入标点
        for i, char in enumerate(text):
            # 先添加字符
            result.append(char)
            
            # 再添加预测的标点（如果有）
            if i < len(predictions) and predictions[i] != 0:  # 不是O标签
                tag = IDX_TO_TAG[predictions[i]]
                
                if tag in TAG_TO_PUNCTUATION:
                    result.append(TAG_TO_PUNCTUATION[tag])
                else:
                    logger.warning(f"标签 {tag} 不在 TAG_TO_PUNCTUATION 映射中，跳过该标签")
        
        return ''.join(result)

    def _post_process_text(self, text):
        """对恢复标点后的文本进行后处理，确保标点符号的一致性"""
        # 检查并替换可能的英文标点
        english_to_chinese = {
            ':': '：',
            ';': '；',
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            '(': '（',
            ')': '）',
            '[': '【',
            ']': '】',
            '{': '｛',
            '}': '｝',
            '...': '……'
        }
        
        for eng, chn in english_to_chinese.items():
            # 保留特殊格式
            if eng == ':' and re.search(r'\d+:\d+', text):
                continue
            text = text.replace(eng, chn)
        
        return text

    def predict_file(self, input_file, output_file):
        """预测文件中的标点"""
        try:
            # 读取输入文件
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # 如果文本为空，返回错误
            if not text or len(text.strip()) == 0:
                logger.error("输入文件为空，无法处理")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("")
                return ""

            # 按行处理模式
            if '\n' in text:
                lines = text.strip().split('\n')
                result_lines = []

                for line in tqdm(lines, desc="处理文本行"):
                    if not line.strip():  # 跳过空行
                        result_lines.append("")
                        continue

                    # 预测单行文本
                    result_line = self.predict_text(line)
                    result_lines.append(result_line)

                # 合并结果
                result_text = '\n'.join(result_lines)
            else:
                # 预测整个文本
                result_text = self.predict_text(text)

            # 写入输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result_text)

            logger.info(f"预测完成，结果已保存至 {output_file}")

            return result_text

        except Exception as e:
            logger.error(f"预测文件时出错: {e}")
            raise

    def predict_text(self, text):
        """预测文本中的标点"""
        try:
            # 预处理文本
            preprocessed_text = self._preprocess_text(text)

            if not preprocessed_text:
                return text  # 返回原始文本

            # 分割成窗口
            windows, char_indices = self._split_text_into_windows(preprocessed_text)

            # 对每个窗口进行预测
            window_predictions = []
            for window in windows:
                valid_positions, predictions = self._predict_window(window)
                window_predictions.append((valid_positions, predictions))

            # 合并窗口预测结果
            merged_predictions = self._merge_window_predictions(preprocessed_text, char_indices, window_predictions)

            # 插入标点
            result_text = self._insert_punctuation(preprocessed_text, merged_predictions)

            # 添加后处理
            result_text = self._post_process_text(result_text)

            return result_text

        except Exception as e:
            logger.error(f"预测文本时出错: {e}")
            return text  # 出错时返回原始文本


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='中文文本标点恢复')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径(无标点)')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径(带标点)')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='模型路径')
    parser.add_argument('--config', type=str, default='configs/config.yml', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，显示更多日志信息')
    args = parser.parse_args()

    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("已启用调试模式")

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件 {args.input} 不存在")
        return

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        # 尝试相对于src目录的路径
        alt_model_path = os.path.join('../', args.model)
        if os.path.exists(alt_model_path):
            args.model = alt_model_path
            logger.info(f"找到模型文件: {args.model}")
        else:
            logger.error(f"模型文件 {args.model} 不存在")
            return

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        # 尝试相对于src目录的路径
        alt_config_path = os.path.join('../', args.config)
        if os.path.exists(alt_config_path):
            args.config = alt_config_path
            logger.info(f"找到配置文件: {args.config}")
        else:
            logger.error(f"配置文件 {args.config} 不存在")
            return

    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建标点恢复器
    restorer = PunctuationRestorer(args.model, args.config)

    # 预测文件
    restorer.predict_file(args.input, args.output)


if __name__ == "__main__":
    main()
