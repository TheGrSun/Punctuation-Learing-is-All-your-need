# Punctuation Learning is All Your Need!

燃尽了，只剩下了雪白的灰，反正也没人用，随便水水吧。

## 项目简介

这是一个基于深度学习的中文标点符号智能预测系统，专门用于解决文本中标点符号缺失或错误的问题。该系统采用BERT预训练模型与双向LSTM和多头注意力机制相结合的架构，能够精确地识别文本中应该插入标点符号的位置。无论是语音识别输出、OCR识别结果还是无标点的原始文本，本系统都能准确地添加标点符号，显著提升文本的可读性和语义清晰度。

## 特性

- **强大的模型架构**：基于预训练中文RoBERTa模型结合BiLSTM与多头注意力机制
- **高效的参数学习**：支持梯度检查点技术和特定层参数冻结优化训练效率
- **混合精度训练**：支持FP16训练加速，提升训练速度并降低显存消耗
- **流式数据处理**：高效处理超大规模数据集，支持增量训练
- **类别不平衡处理**：采用Focal Loss与加权交叉熵联合损失函数优化稀有标点符号预测
- **自适应学习率**：余弦退火学习率调度与预热策略
- **丰富的标点支持**：支持逗号、句号、问号、感叹号、冒号等22种中文标点
- **窗口滑动预测**：使用重叠窗口技术处理长文本，确保上下文连贯性
- **TensorBoard可视化**：实时监控训练过程中的各项指标

## 环境要求
- pip install requirements.txt

## 项目结构

```
├── configs/                # 配置文件目录
│   └── config.yml          # 主配置文件
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   ├── test/               # 测试数据
│   └── processed/          # 处理后的数据
├── models/                 # 保存的模型
│   ├── best_model.pth      # 最佳性能模型
│   ├── latest_model.pth    # 最新训练模型
│   └── initial_model.pth   # 初始模型
├── src/                    # 源代码
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练代码
│   ├── predict.py          # 预测代码
│   ├── evaluate.py         # 评估代码
│   ├── process.py          # 数据处理
│   ├── datacheck.py        # 数据检查
│   ├── data_merge.py       # 数据合并
│   ├── utils.py            # 工具函数
│   ├── __init__.py         # 包初始化
├── BERT_models/            # 预训练的BERT模型
│   └── hfl/
│       └── chinese-roberta-wwm-ext/ # 中文RoBERTa模型
├── cache/               # 数据权重缓存
│   ├── preprocessed/    # 预处理数据缓存
│   └── class_weights.pkl # 类别权重缓存
├── runs/                # TensorBoard日志
│   └── run_YYYYMMDD_HHMMSS/ # 训练运行记录
├── requirements.txt     # 依赖包
```

## 模型架构

本项目采用多层级融合的深度学习架构：

1. **BERT编码层**：利用预训练的中文RoBERTa模型提取文本的上下文语义表示
2. **双向LSTM层**：捕获长距离序列依赖关系，增强对标点位置的感知能力
3. **多头注意力层**：进一步强化重要上下文信息的权重，提升预测准确率
4. **残差连接与层归一化**：优化梯度流动，提高模型训练稳定性
5. **全连接分类层**：将特征映射到22种不同标点类型的概率分布

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据预处理

```bash
python -m src.process
```

处理参数可在`configs/config.yml`的`process`部分配置。

### 训练模型

```bash
python -m src.train
```

训练参数可在`configs/config.yml`的`training`部分配置。

### 模型评估

```bash
python -m src.evaluate
```

### 标点预测

单个文件预测：
```bash
python -m src.predict --input 无标点文件.txt --output 带标点文件.txt --model models/best_model.pth
```

命令行参数说明：
- `--input`：输入文件路径(无标点文本)
- `--output`：输出文件路径(带标点文本)
- `--model`：模型路径，默认为`models/best_model.pth`
- `--config`：配置文件路径，默认为`configs/config.yml`
- `--debug`：启用调试模式，显示更多日志信息

## 配置说明

在`configs/config.yml`中可配置以下参数：

### 模型配置

```yaml
model:
  max_seq_length: 256  # 序列最大长度
  num_tags: 22         # 标点类型数量
  dropout: 0.15        # Dropout比例
  bert_model_path: 'BERT_models/hfl/chinese-roberta-wwm-ext'  # BERT模型路径
  
  # BERT、LSTM、Attention等参数配置
```

### 训练配置

```yaml
training:
  batch_size: 256       # 批次大小
  learning_rate: 2e-5   # 学习率
  epochs: 200           # 训练轮次
  gradient_clip_val: 1.0 # 梯度裁剪值
  patience: 10          # 早停耐心值
  fp16: true            # 是否使用FP16混合精度
  
  # 流式数据处理、学习率调度器等配置
```

### 数据处理配置

```yaml
process:
  chunk_size: 100000   # 数据块大小
  train_ratio: 0.8     # 训练集比例
  
  # 其他数据处理参数
```

## 支持的标点类型

本系统支持以下标点符号：

- 单字符标点：逗号(，)、句号(。)、问号(？)、感叹号(！)、分号(；)、冒号(：)、顿号(、)
- 成对标点：双引号(“ ”)、单引号(‘ ’)、括号(（）)、书名号(《》)、大括号({})、中括号(【】)
- 特殊标点：省略号(……)、破折号(——)

## 性能指标

在ThuCnews数据集上的性能：
- 准确率：>97%
- 精确率：>52%
- 召回率：>54%
- F1分数：>57%

PS：由于数据集的特殊性，本项目在实际应用中可能存在一定的局限性，并且由于该任务目标的特性，非标点字符占绝对多数且准确率特别高，所以实际上模型性能应该具体到各个标点符号的表现。

## 贡献

欢迎贡献代码、改进模型架构或提出改进建议。请在GitHub上提交问题或拉取请求。

## 引用

如果您在研究中使用了本项目，请引用：

```
@misc{PunctuationLearning,
  author = {Aixinkakula},
  title = {Punctuation Learning is All Your Need},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/TheGrSun/Punctuation-Learing-is-All-your-need}
}
```

## 许可证

本项目采用 MIT 许可证
