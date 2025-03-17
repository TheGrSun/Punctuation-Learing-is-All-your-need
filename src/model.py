import torch
import torch.nn as nn
from transformers import BertModel
import yaml
import os
import logging


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 保存配置
        self.config = config
        
        # 混合精度训练设置
        self.use_amp = config['training'].get('fp16', False)
        
        # BERT模型
        cache_dir = config['model'].get('cache_dir', None)

        self.bert = BertModel.from_pretrained(
            config['model']['bert_model_path'], 
            output_hidden_states=True,
            cache_dir=cache_dir
        )

        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True
        
        # 启用梯度检查点以节省显存
        if hasattr(self.bert, 'gradient_checkpointing_enable'):
            self.bert.gradient_checkpointing_enable()

            # 优化梯度钩子函数
            def make_inputs_require_grad(module, input, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
                else:
                    output[0].requires_grad_(True)
            self.bert.embeddings.register_forward_hook(make_inputs_require_grad)
        
        # 冻结BERT底层参数
        if config.get('training', {}).get('freeze_bert_layers', 0) > 0:
            freeze_layers = config['training']['freeze_bert_layers']
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
            logging.info(f"已冻结BERT的嵌入层和前{freeze_layers}个编码层")
        
        # Dropout层 - BERT输出后
        self.bert_dropout = nn.Dropout(config['model']['dropout'])

        # BiLSTM层
        lstm_config = config['model']['lstm']
        self.lstm = nn.LSTM(input_size=lstm_config['input_size'],
                            hidden_size=lstm_config['hidden_size'],
                            num_layers=lstm_config['num_layers'],
                            bidirectional=lstm_config['bidirectional'],
                            batch_first=lstm_config['batch_first'],
                            dropout=lstm_config['dropout'] if lstm_config['num_layers'] > 1 else 0)
        
        # 层归一化 - LSTM输出后
        lstm_output_size = lstm_config['hidden_size'] * 2 if lstm_config['bidirectional'] else lstm_config['hidden_size']
        self.lstm_layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Dropout层 - LSTM输出后
        self.lstm_dropout = nn.Dropout(config['model']['dropout'])

        # 多头自注意力层
        attention_config = config['model']['attention']
        self.attention = nn.MultiheadAttention(embed_dim=attention_config['embed_dim'],
                                               num_heads=attention_config['num_heads'],
                                               dropout=attention_config['dropout'],
                                               batch_first=attention_config['batch_first'])
        
        # 层归一化 - 注意力输出后
        self.attention_layer_norm = nn.LayerNorm(attention_config['embed_dim'])
        
        # Dropout层 - 注意力输出后
        self.attention_dropout = nn.Dropout(config['model']['dropout'])

        # 全连接层 - 直接使用注意力层的嵌入维度作为输入
        fc_input_size = attention_config['embed_dim']
        fc_output_size = config['model']['num_tags']
        self.classifier = nn.Linear(fc_input_size, fc_output_size)

        # 添加sigmoid激活
        self.sigmoid = nn.Sigmoid()
        
        # 初始化模型权重
        self._init_weights()

    def _init_weights(self):
        """初始化非BERT部分的模型权重"""
        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)  # 正交初始化对RNN/LSTM有效
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # 初始化全连接层
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        
        # 初始化层归一化
        for module in [self.lstm_layer_norm, self.attention_layer_norm]:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, input_ids, attention_mask, tags=None, mask=None):
        # BERT特征提取 - 优化内存使用
        with torch.set_grad_enabled(not self.bert.training or torch.is_grad_enabled()):
            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=self.use_amp):
                # 使用return_dict=False以减少内存开销
                bert_outputs = self.bert(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    return_dict=False
                )
                sequence_output = bert_outputs[0]  # 最后一层的隐藏状态
                
                # 应用dropout
                sequence_output = self.bert_dropout(sequence_output)

                # LSTM特征提取
                lstm_output, _ = self.lstm(sequence_output)
                
                # 应用层归一化和dropout
                lstm_output = self.lstm_layer_norm(lstm_output)
                lstm_output = self.lstm_dropout(lstm_output)

                # 多头自注意力特征提取 - 使用key_padding_mask优化计算
                key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
                attn_output, _ = self.attention(
                    lstm_output, 
                    lstm_output, 
                    lstm_output, 
                    key_padding_mask=key_padding_mask,
                    need_weights=False  # 设置need_weights=False以节省计算
                )
                
                # 残差连接 + 层归一化
                combined_output = lstm_output + attn_output
                combined_output = self.attention_layer_norm(combined_output)
                combined_output = self.attention_dropout(combined_output)

                # 分类层
                logits = self.classifier(combined_output)
        
        # 应用sigmoid激活
        probs = self.sigmoid(logits)
        
        # 计算多标签损失 - 与训练脚本兼容
        loss = None
        if tags is not None and mask is not None:
            # 注意：训练脚本中使用BCEWithLogitsLoss并传入pos_weight参数
            # 这里我们只计算基本损失，让训练脚本处理权重
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            # 计算每个位置的损失
            element_loss = loss_fn(logits, tags.float())
            # 应用mask
            active_loss = mask.unsqueeze(-1).expand_as(element_loss)
            # 计算平均损失
            loss = torch.sum(element_loss * active_loss) / (torch.sum(active_loss) + 1e-10)
        
        return {
            'logits': logits,
            'probs': probs,
            'loss': loss,
            'predictions': (probs > 0.5).long()  # 阈值预测
        }


# 模型保存函数
def save_model(model, path, optimizer=None, scheduler=None, epoch=None, best_f1=None):
    """保存模型到指定路径，支持保存训练状态"""
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 准备保存内容
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': model.config if hasattr(model, 'config') else None
    }
    
    # 如果提供了优化器，保存优化器状态
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    # 如果提供了调度器，保存调度器状态
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存训练状态
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    if best_f1 is not None:
        save_dict['best_f1'] = best_f1
    
    # 保存模型
    torch.save(save_dict, path)
    logging.info(f"模型已保存到 {path}")


# 模型加载函数
def load_model(path, config=None, device='cpu', load_optimizer=False, load_scheduler=False):
    """从指定路径加载模型，支持加载训练状态"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件 {path} 不存在")
    
    # 加载模型状态
    checkpoint = torch.load(path, map_location=torch.device(device))
    
    # 如果没有提供配置，尝试从检查点加载
    if config is None and 'config' in checkpoint:
        config = checkpoint['config']
    
    if config is None:
        raise ValueError("需要提供配置或检查点中包含配置")
    
    # 创建新模型
    model = Model(config)
    
    # 加载模型状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 准备返回结果
    result = {'model': model}
    
    # 如果需要加载优化器
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    # 如果需要加载调度器
    if load_scheduler and 'scheduler_state_dict' in checkpoint:
        result['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
    
    # 加载训练状态
    if 'epoch' in checkpoint:
        result['epoch'] = checkpoint['epoch']
    
    if 'best_f1' in checkpoint:
        result['best_f1'] = checkpoint['best_f1']
    
    return result

if __name__ == "__main__":
    # 从配置文件加载配置
    with open('configs/config.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建模型
    model = Model(config)
    print(model)
