import os
import torch
import numpy as np
import yaml
import logging
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import json

from model import load_model
from train import StreamingPunctuationDataset, collate_fn
from utils import IDX_TO_TAG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, data_loader, device, num_tags, max_steps=None):
    """评估模型性能，返回每个标点符号的详细指标"""
    model.eval()

    # 初始化每个标签的预测和真实标签列表
    tag_predictions = {idx: [] for idx in range(num_tags)}
    tag_labels = {idx: [] for idx in range(num_tags)}
    
    # 所有预测和标签
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="评估")):
            # 如果设置了最大步数，则限制评估步数
            if max_steps is not None and batch_idx >= max_steps:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            # 检查outputs的结构并正确提取logits
            if isinstance(outputs, dict):
                # 处理Model类返回的字典格式
                logits = outputs['logits']
            elif isinstance(outputs, tuple) and len(outputs) >= 2:
                # 处理PunctuationModel类返回的元组格式
                logits = outputs[1]
            else:
                # 处理其他情况
                logits = outputs[0]
                
            predictions = torch.argmax(logits, dim=-1)

            # 收集非padding位置的预测和标签
            active_positions = batch["attention_mask"].view(-1) == 1
            active_preds = predictions.view(-1)[active_positions]
            active_labels = batch["labels"].view(-1)[active_positions]

            # 将预测和标签添加到总列表
            all_predictions.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())
            
            # 为每个标签收集预测和真实标签
            for idx in range(num_tags):
                # 找到真实标签为idx的位置
                idx_positions = (active_labels == idx)
                if idx_positions.sum() > 0:
                    # 收集这些位置的预测
                    tag_predictions[idx].extend(active_preds[idx_positions].cpu().numpy())
                    tag_labels[idx].extend(active_labels[idx_positions].cpu().numpy())
                
                # 找到预测为idx的位置
                pred_idx_positions = (active_preds == idx)
                if pred_idx_positions.sum() > 0:
                    # 确保这些预测对应的真实标签被收集
                    if idx not in tag_labels:
                        tag_labels[idx] = []
                    tag_labels[idx].extend(active_labels[pred_idx_positions].cpu().numpy())
            
            # 定期释放缓存，减少内存占用
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 计算总体指标
    overall_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    overall_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    overall_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    overall_accuracy = accuracy_score(all_labels, all_predictions)

    # 生成详细报告
    report = classification_report(
        all_labels, all_predictions,
        labels=list(range(num_tags)),
        target_names=list(IDX_TO_TAG.values()),
        zero_division=0,
        output_dict=True
    )

    # 计算每个标点符号的指标
    tag_metrics = {}
    for idx in range(num_tags):
        tag_name = IDX_TO_TAG[idx]
        
        # 从报告中获取指标
        if tag_name in report:
            tag_metrics[tag_name] = {
                'precision': report[tag_name]['precision'],
                'recall': report[tag_name]['recall'],
                'f1-score': report[tag_name]['f1-score'],
                'support': report[tag_name]['support']
            }
        else:
            # 如果报告中没有该标签（可能是因为测试集中没有该标签的样本）
            tag_metrics[tag_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1-score': 0.0,
                'support': 0
            }

    # 返回总体指标和每个标点符号的指标
    metrics = {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'accuracy': overall_accuracy,
        },
        'tag_metrics': tag_metrics,
        'report': report
    }

    return metrics


def visualize_metrics(metrics, output_dir):
    """可视化评估指标"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取每个标签的指标
    tag_metrics = metrics['tag_metrics']
    
    # 创建DataFrame用于可视化
    data = []
    for tag, tag_metric in tag_metrics.items():
        data.append({
            'Tag': tag,
            'Precision': tag_metric['precision'],
            'Recall': tag_metric['recall'],
            'F1-Score': tag_metric['f1-score'],
            'Support': tag_metric['support']
        })
    
    df = pd.DataFrame(data)
    
    # 按F1分数排序
    df = df.sort_values(by='F1-Score', ascending=False)
    
    # 设置中文字体
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
    except:
        font = None
        logger.warning("无法加载中文字体，图表中的中文可能显示不正确")
    
    # 绘制F1分数条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='F1-Score', y='Tag', data=df)
    plt.title('各标点符号的F1分数', fontproperties=font if font else None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_scores.png'), dpi=300)
    
    # 绘制精确率和召回率对比图
    plt.figure(figsize=(12, 8))
    df_melted = pd.melt(df, id_vars=['Tag'], value_vars=['Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Value')
    sns.barplot(x='Value', y='Tag', hue='Metric', data=df_melted)
    plt.title('各标点符号的精确率、召回率和F1分数', fontproperties=font if font else None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_f1.png'), dpi=300)
    
    # 绘制支持度（样本数量）条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Support', y='Tag', data=df)
    plt.title('各标点符号的样本数量', fontproperties=font if font else None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'support.png'), dpi=300)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    heatmap_data = df[['Precision', 'Recall', 'F1-Score']].values
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=['Precision', 'Recall', 'F1-Score'],
                yticklabels=df['Tag'])
    plt.title('各标点符号的性能指标热力图', fontproperties=font if font else None)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300)
    
    logger.info(f"可视化结果已保存到 {output_dir} 目录")


def generate_confusion_matrix(all_labels, all_predictions, num_tags, output_dir):
    """生成混淆矩阵并可视化"""
    from sklearn.metrics import confusion_matrix
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions, labels=range(num_tags))
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除以零的情况
    
    # 绘制混淆矩阵
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=[IDX_TO_TAG[i] for i in range(num_tags)],
                yticklabels=[IDX_TO_TAG[i] for i in range(num_tags)])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('标点符号预测的归一化混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    
    logger.info(f"混淆矩阵已保存到 {output_dir} 目录")


def save_metrics_to_file(metrics, output_path):
    """将评估指标保存到文件"""
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将指标转换为可序列化的格式
    serializable_metrics = {
        'overall': metrics['overall'],
        'tag_metrics': metrics['tag_metrics']
    }
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=4)
    
    logger.info(f"评估指标已保存到 {output_path}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='评估标点符号预测模型')
    parser.add_argument('--config', type=str, default='configs/config.yml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='模型文件路径')
    parser.add_argument('--data_file', type=str, default=None, help='评估数据文件路径，默认使用配置文件中的测试集')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='评估结果输出目录')
    parser.add_argument('--max_steps', type=int, default=None, help='最大评估步数，用于流式数据集')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小，默认使用配置文件中的设置')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化分词器
    tokenizer = BertTokenizerFast.from_pretrained(config['model']['bert_model_path'])
    
    # 确定评估数据文件
    data_file = args.data_file if args.data_file else config['data']['test_file']
    logger.info(f"使用评估数据文件: {data_file}")
    
    # 确定批次大小
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    
    # 加载数据集
    logger.info("加载评估数据集...")
    dataset = StreamingPunctuationDataset(
        data_file,
        tokenizer,
        config['model']['max_seq_length'],
        cache_dir=config['data']['cache_dir'],
        use_multiprocessing=True,
        buffer_size=config['training'].get('buffer_size', 1000),
        calculate_stats=True,
        shuffle=False
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['training'].get('dataloader_workers', 2),
        pin_memory=True
    )
    
    # 加载模型
    logger.info(f"从 {args.model_path} 加载模型...")
    result = load_model(args.model_path, config, device)
    model = result['model']
    model.to(device)
    model.eval()
    
    # 评估模型
    logger.info("开始评估模型...")
    metrics = evaluate_model(model, data_loader, device, config['model']['num_tags'], args.max_steps)
    
    # 打印总体评估结果
    logger.info(f"总体评估结果:")
    logger.info(f"精确率: {metrics['overall']['precision']:.4f}")
    logger.info(f"召回率: {metrics['overall']['recall']:.4f}")
    logger.info(f"F1分数: {metrics['overall']['f1']:.4f}")
    logger.info(f"准确率: {metrics['overall']['accuracy']:.4f}")
    
    # 打印每个标点符号的评估结果
    logger.info("各标点符号的评估结果:")
    for tag, tag_metric in metrics['tag_metrics'].items():
        logger.info(f"{tag}: 精确率={tag_metric['precision']:.4f}, 召回率={tag_metric['recall']:.4f}, F1={tag_metric['f1-score']:.4f}, 样本数={tag_metric['support']}")
    
    # 保存评估指标
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_metrics_to_file(metrics, metrics_path)
    
    # 可视化评估结果
    logger.info("生成评估结果可视化...")
    visualize_metrics(metrics, args.output_dir)
    
    logger.info(f"评估完成，结果已保存到 {args.output_dir} 目录")


if __name__ == "__main__":
    main()
