import torch
import time
import logging
from typing import Optional, Tuple, Dict
import numpy as np
from torch.cuda.amp import autocast
import torch.nn as nn
import torchvision.utils as vutils
from core.models.i2u_net import I2U_Net, NonLocal_spp_inception_block
import os
from datetime import timedelta

def setup_logger(log_dir: str, name: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器
    """
    name = name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def find_optimal_batch_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    max_batch_size: int = 32,
    step_size: int = 4
) -> int:
    """
    查找最优批次大小
    """
    batch_size = max_batch_size
    while batch_size > step_size:
        try:
            with autocast():
                sample = sample_input[:batch_size].cuda()
                _ = model(sample)
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size -= step_size
            else:
                raise e
    return step_size

def safe_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> torch.Tensor:
    """
    安全的前向传播，处理OOM错误
    """
    try:
        # 确保输入是4D的 [B, C, H, W]
        if inputs.dim() == 3:  # 如果是 [C, H, W]
            inputs = inputs.unsqueeze(0)  # 添加批次维度 [1, C, H, W]
            
        # 根据scaler是否为None决定是否使用混合精度
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
        else:
            outputs = model(inputs)
        return outputs
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return safe_forward(model, inputs, scaler)
        raise e

class Timer:
    """
    训练时间估计器
    """
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
        
    def epoch_complete(self, epoch: int) -> Dict[str, float]:
        """
        记录每个epoch的完成时间并估计剩余时间
        """
        current_time = time.time()
        epoch_time = current_time - self.start_time if not self.epoch_times else current_time - self.epoch_times[-1]
        self.epoch_times.append(current_time)
        
        # 计算平均epoch时间
        avg_epoch_time = np.mean(self.epoch_times[-5:]) if len(self.epoch_times) > 5 else np.mean(self.epoch_times)
        
        # 估计剩余时间
        remaining_epochs = self.total_epochs - (epoch + 1)
        estimated_remaining = remaining_epochs * avg_epoch_time
        
        return {
            'epoch_time': epoch_time,
            'avg_epoch_time': avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'total_elapsed': current_time - self.start_time
        }

def visualize_features(
    writer,
    model: nn.Module,
    inputs: torch.Tensor,
    epoch: int,
    prefix: str = 'train'
) -> None:
    """可视化模型的特征图和注意力图
    
    Args:
        writer: TensorBoard写入器
        model: 模型
        inputs: 输入张量
        epoch: 当前轮次
        prefix: 前缀
    """
    if not hasattr(writer, 'add_image'):
        return  # 如果writer不支持add_image，直接返回
        
    # 保存原始训练状态
    training = model.training
    
    # 设置为评估模式
    model.eval()
    
    # 获取特征图
    features = []
    
    def hook(module, input, output):
        features.append(output.detach().cpu())
    
    hooks = []
    # 为每个卷积层注册钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'encoder' in name:
            hooks.append(module.register_forward_hook(hook))
    
    with torch.no_grad():
        try:
            _ = model(inputs)
            
            # 移除钩子
            for h in hooks:
                h.remove()
            
            # 可视化特征
            for i, feat in enumerate(features):
                if i % 2 == 0:  # 只可视化部分层
                    try:
                        # 特征图
                        b, c, h, w = feat.shape
                        if c > 64:  # 如果通道数太多，只显示前64个
                            feat = feat[:, :64]
                        
                        # 使用make_grid创建网格图像
                        grid = vutils.make_grid(
                            feat[0].unsqueeze(1),  # 为每个特征通道添加通道维度
                            nrow=8,  # 每行显示8个特征图
                            normalize=True,  # 将值归一化到[0,1]范围
                            scale_each=True,  # 对每个特征单独归一化
                            pad_value=0.5  # 使用灰色作为填充
                        )
                        
                        # 确保兼容TensorBoard
                        grid = ensure_compatible_tensor(grid)
                        
                        # 保存到TensorBoard
                        writer.add_image(f'{prefix}/features_layer_{i}', grid, epoch)
                        
                        # 注意力图可视化
                        if isinstance(model.module if hasattr(model, 'module') else model, I2U_Net) and \
                           hasattr(model.module if hasattr(model, 'module') else model, f'hifa{i//2+1}'):
                            try:
                                # 生成注意力图
                                feat_reshaped = feat[0].view(c, h * w)
                                
                                # 限制计算规模，避免注意力图过大
                                max_size = min(64, feat_reshaped.size(0))
                                feat_reshaped = feat_reshaped[:max_size, :]
                                
                                # 确保不超过最大内存限制
                                if feat_reshaped.size(1) > 10000:  # 限制最大特征图尺寸
                                    stride = feat_reshaped.size(1) // 10000 + 1
                                    feat_reshaped = feat_reshaped[:, ::stride]
                                
                                # 计算注意力
                                attention = torch.matmul(feat_reshaped, feat_reshaped.transpose(0, 1))
                                attention = torch.softmax(attention, dim=1)
                                
                                # 创建网格图像
                                attention_grid = vutils.make_grid(
                                    attention.unsqueeze(1).unsqueeze(1),
                                    nrow=8,
                                    normalize=True,
                                    scale_each=True
                                )
                                
                                # 确保兼容TensorBoard
                                attention_grid = ensure_compatible_tensor(attention_grid)
                                
                                # 保存到TensorBoard
                                writer.add_image(f'{prefix}/attention_layer_{i}', attention_grid, epoch)
                            except Exception as e:
                                print(f"注意力图可视化错误: {e}")
                    except Exception as e:
                        print(f"特征可视化错误: {e}")
        except Exception as e:
            print(f"特征提取错误: {e}")
            # 确保所有钩子被移除
            for h in hooks:
                h.remove()
    
    # 恢复训练状态
    if training:
        model.train()

def ensure_compatible_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """确保张量格式与TensorBoard兼容
    
    转换张量为可以被TensorBoard显示的格式（通常是3通道RGB图像）
    
    Args:
        tensor: 输入张量
        
    Returns:
        转换后的张量，保证与TensorBoard兼容
    """
    # 确保张量在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 确保张量值在[0,1]范围内
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)
    
    # 处理不同的张量维度情况
    if tensor.dim() == 2:  # [H,W]
        tensor = tensor.unsqueeze(0)  # 添加通道维度 [1,H,W]
    
    if tensor.dim() == 3:  # [C,H,W]
        if tensor.size(0) != 3 and tensor.size(0) != 1:
            # 如果通道数既不是1也不是3，选择第一个通道
            tensor = tensor[0].unsqueeze(0)  # [1,H,W]
    
    if tensor.dim() == 4:  # [B,C,H,W]
        tensor = tensor[0]  # 选择第一个样本 [C,H,W]
        if tensor.size(0) != 3 and tensor.size(0) != 1:
            # 如果通道数既不是1也不是3，选择第一个通道
            tensor = tensor[0].unsqueeze(0)  # [1,H,W]
    
    # 将单通道转换为三通道RGB
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    
    return tensor 

def monitor_gradient_stats(model, writer, epoch, step=0, n_layers_to_log=5):
    """监控梯度统计信息，检测潜在的梯度爆炸或消失问题"""
    # 存储梯度统计信息
    grad_stats = {}
    
    # 获取梯度
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 计算范数
            norm = param.grad.data.norm(2).item()
            if not np.isfinite(norm):  # 检查NaN或Inf
                grad_stats[f'grad_norm/{name}'] = float('nan')
            else:
                grad_stats[f'grad_norm/{name}'] = norm
            
            # 统计梯度最大/最小值
            if torch.isfinite(param.grad.data).all():
                grad_stats[f'grad_max/{name}'] = param.grad.data.max().item()
                grad_stats[f'grad_min/{name}'] = param.grad.data.min().item()
                grad_stats[f'grad_mean/{name}'] = param.grad.data.mean().item()
                grad_stats[f'grad_std/{name}'] = param.grad.data.std().item()
                grad_norms[name] = norm
    
    # 选择梯度范数最大的几个层进行记录
    largest_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:n_layers_to_log]
    
    for name, norm in largest_grads:
        writer.add_scalar(f'grad_tracking/{name}', norm, epoch * 1000 + step)
    
    # 计算所有层梯度范数的均值和标准差
    valid_norms = [norm for norm in grad_norms.values() if np.isfinite(norm)]
    if valid_norms:
        writer.add_scalar('grad_stats/mean_norm', np.mean(valid_norms), epoch * 1000 + step)
        writer.add_scalar('grad_stats/max_norm', np.max(valid_norms), epoch * 1000 + step)
        
    return grad_stats, largest_grads 