from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class TrainingConfig:
    # 基础训练参数
    model_type: str = 'i2u_net'
    n_epoch: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    
    # 训练策略
    use_amp: bool = False  # 默认禁用混合精度训练，与原始项目保持一致
    clip_grad: float = 0.1
    class_weights: bool = True
    
    # 数据相关
    per_val: float = 0.1
    aug: bool = True
    num_workers: int = 4  # 与原始项目一致
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu: bool = True
    
    # 检查点配置
    resume: Optional[str] = None
    save_freq: int = 10
    save_best: bool = True  # 是否保存最佳模型
    
    # 日志配置
    log_dir: str = 'runs'
    log_freq: int = 20
    log_attention: bool = False  # 是否记录和可视化注意力图
    
    # 新增选项
    enable_visualization: bool = True  # 是否启用特征可视化
    use_original_optim: bool = True    # 是否使用原始项目优化器设置
    use_original_sampler: bool = True  # 是否使用原始项目采样器策略

@dataclass
class ModelConfig:
    # 模型架构参数
    in_channels: int = 1
    num_classes: int = 6
    pretrained: bool = True
    
    # I2U-Net特定参数
    filters: List[int] = (64, 128, 256, 512)
    hifa_ratio: int = 2
    eca_kernel_size: int = 3
    
    # 解码器参数
    decoder_channels: int = 32
    use_attention: bool = True

@dataclass
class OptimConfig:
    # 优化器配置
    optimizer: str = 'adamw'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # 学习率调度器配置
    scheduler: str = 'cosine'
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    
    # 损失函数权重
    class_weights: List[float] = (0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852)

def get_config(args):
    """
    从命令行参数创建配置对象
    """
    train_cfg = TrainingConfig(
        model_type=args.arch,
        n_epoch=args.n_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_grad=args.clip,
        per_val=args.per_val,
        aug=args.aug,
        class_weights=args.class_weights,
        resume=args.resume,
        # 新增配置项
        use_amp=not args.disable_amp,
        use_original_optim=args.use_original_optim,
        use_original_sampler=args.use_original_sampler,
        log_attention=args.log_attention if hasattr(args, 'log_attention') else False,
        save_best=not args.no_save_best if hasattr(args, 'no_save_best') else True,
        enable_visualization=not args.no_visualization if hasattr(args, 'no_visualization') else True
    )
    
    model_cfg = ModelConfig()
    optim_cfg = OptimConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    return train_cfg, model_cfg, optim_cfg 