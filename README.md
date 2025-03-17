# Punctuation Learning is All Your Need!

长路漫漫，我心悠悠。

实际上应该是GPU is All Your Need

痛斥AutoDL，你这么多卡怎么我每天都要抢？

## 项目简介

这是一个基于深度学习的中文标点符号预测模型，用以处理给无标点文本中加标点的任务。

该模型采用BERT预训练模型与双向LSTM和多头注意力机制相结合的架构，能够高效的识别无标点文本中应该插入标点符号的位置和类型。

可以给ASR处理后的无标点文本加标点或者用于古文本修复，虽然没有LLM好用就是了。

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

参见requirements.txt文件

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
pip install requirements.txt
```

### 数据预处理

```bash
python src/process
```

处理参数可在`configs/config.yml`的`process`部分配置。

### 训练模型

```bash
python src/train
```

训练参数可在`configs/config.yml`的`training`部分配置。

### 模型评估

```bash
python src/evaluate.py --model_path models/best_model.pth --output_dir evaluation_results
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
作者使用ThuCnews数据集进行模型的训练，数据集地址为:http://thuctc.thunlp.org/

数据集简介:
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

在该数据集上训练的模型性能指标如下：
- 准确率：>97%
- 精确率：>52%
- 召回率：>54%
- F1分数：>57%

PS：由于数据集的特殊性，该模型在实际应用中可能存在一定的局限性，对于新闻类的文本或许效果还不错，若使用小说之类的文本进行预测的话就可能效果会变差一些，并且由于该任务目标的特性，非标点字符占数据集的绝对多数，而稀有标签甚至可能不出现或只出现极少数，所以总体的准确率会很高，而其他指标会偏低，因此实际上模型性能应该具体到各个标点符号的指标表现。

## 贡献

欢迎贡献代码、改进模型架构或提出改进建议。请在GitHub上提交问题或拉取请求。

如果对该项目有任何问题或者需要训练好的模型或数据集，请在GitHub上拉取请求。


## To do list
-添加英文标点映射和相应处理逻辑（或许会做）

-添加日语标点映射和相应处理逻辑（或许会做）

-改进对于成对标点符号（如双引号，单引号，书名号等）的预测结果（马上会做）

-改进对于稀有标签（如破折号，大括号，中括号等）的预测结果（马上会做）

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
