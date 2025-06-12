import argparse
import os
from datetime import datetime
from os.path import join as pjoin
from ast import arg
import sys
import io
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from torch.cuda.amp import autocast

import core.loss
import torchvision.utils as vutils
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
from core.loader.data_loader import *
from core.metrics import runningScore
from core.models import get_model
from core.utils import np_to_tb
from core.utils.training_utils import (
    setup_logger, find_optimal_batch_size, safe_forward,
    Timer, visualize_features, monitor_gradient_stats
)
import time
from core.config import get_config
from torch.cuda.amp import GradScaler
import logging

# 配置CUDNN，尝试解决CUDNN警告
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# 创建全局logger和日志分离配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加控制台处理器，确保日志输出到控制台
console_handler = logging.StreamHandler(sys.stderr)  # 使用stderr，避免与tqdm冲突（tqdm使用stdout）
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Fix the random seeds: 
torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


def split_train_val(args, per_val=0.1):
    # create inline and crossline sections for training and validation:
    loader_type = 'section'
    labels = np.load(pjoin('data', 'train', 'train_labels.npy'))
    i_list = list(range(labels.shape[0]))
    i_list = ['i_'+str(inline) for inline in i_list]

    x_list = list(range(labels.shape[1]))
    x_list = ['x_'+str(crossline) for crossline in x_list]

    list_train_val = i_list + x_list

    # create train and test splits:
    list_train, list_val = train_test_split(
        list_train_val, test_size=per_val, shuffle=True)

    # write to files to disK:
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(
        pjoin('data', 'splits', loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(pjoin('data', 'splits', loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()


# 添加原始项目的采样器
class OriginalCustomSamplerTrain(torch.utils.data.Sampler):
    def __init__(self, train_list):
        self.train_list = train_list
        
    def __iter__(self):
        char = ['i' if np.random.randint(2) == 1 else 'x']
        self.indices = [idx for (idx, name) in enumerate(
            self.train_list) if char[0] in name]
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    
    def __len__(self):
        return len(self.train_list)

class OriginalCustomSamplerVal(torch.utils.data.Sampler):
    def __init__(self, val_list):
        self.val_list = val_list
        
    def __iter__(self):
        char = ['i' if np.random.randint(2) == 1 else 'x']
        self.indices = [idx for (idx, name) in enumerate(
            self.val_list) if char[0] in name]
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    
    def __len__(self):
        return len(self.val_list)

def visualize_attention(attention_map, size_required):
    try:
        # 添加形状检查和调整
        attention_map = attention_map.detach().cpu()
        b, c, h, w = size_required
        
        # 如果是一维注意力图，尝试重塑为二维
        if len(attention_map.shape) == 2:
            total_elements = attention_map.numel()
            if total_elements == b * h * w:
                attention_map = attention_map.view(b, 1, h, w)
            else:
                # 记录日志但不报错
                logger.warning(f"注意力图形状 {attention_map.shape} 无法重塑为 {size_required}")
                return None
        
        # 确保大小匹配
        if attention_map.shape != size_required:
            attention_map = F.interpolate(attention_map, size=(h, w), mode='bilinear', align_corners=False)
            
        return attention_map
    except Exception as e:
        logger.warning(f"注意力图可视化错误: {str(e)}")
        return None

def train(args):
    # 首先生成训练集和验证集的分割
    if not os.path.exists(pjoin('data', 'splits', 'section_train.txt')):
        print("正在生成训练集和验证集的分割文件...")
        split_train_val(args, per_val=args.per_val)
        print("分割文件生成完成！")

    # 获取配置
    train_cfg, model_cfg, optim_cfg = get_config(args)
    
    # 处理命令行参数
    if args.no_visualization:
        train_cfg.enable_visualization = False
        print("特征可视化已禁用")
    
    # 设置设备
    device = torch.device(train_cfg.device)
    
    # 设置日志
    current_time = datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join(train_cfg.log_dir, current_time + f"_{train_cfg.model_type}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 移除现有的文件处理器（如果有的话）以避免重复
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 设置训练计时器
    timer = Timer(train_cfg.n_epoch)
    
    # 设置混合精度训练
    scaler = GradScaler() if train_cfg.use_amp else None
    
    # 数据加载和增强设置
    if train_cfg.aug:
        data_aug = Compose([
            RandomRotate(10),
            RandomHorizontallyFlip(),
            AddNoise()
        ])
    else:
        data_aug = None

    # 获取训练和验证数据列表
    train_list = []
    val_list = []
    with open(pjoin('data', 'splits', 'section_train.txt'), 'r') as f:
        train_list = [line.strip() for line in f.readlines()]
    with open(pjoin('data', 'splits', 'section_val.txt'), 'r') as f:
        val_list = [line.strip() for line in f.readlines()]

    train_set = section_loader(
        is_transform=True,
        split='train',
        augmentations=data_aug
    )
    val_set = section_loader(
        is_transform=True,
        split='val'
    )

    # 创建数据加载器 - 根据配置选择采样器
    if train_cfg.use_original_sampler:
        logger.info("使用原始项目的采样器策略")
        trainloader = data.DataLoader(
            train_set,
            batch_size=train_cfg.batch_size if train_cfg.batch_size else 8,
            sampler=OriginalCustomSamplerTrain(train_list),
            num_workers=train_cfg.num_workers
        )
        valloader = data.DataLoader(
            val_set,
            batch_size=train_cfg.batch_size if train_cfg.batch_size else 8,
            sampler=OriginalCustomSamplerVal(val_list),
            num_workers=train_cfg.num_workers
        )
    else:
        trainloader = data.DataLoader(
            train_set,
            batch_size=train_cfg.batch_size,
            sampler=CustomSamplerTrain(train_list),
            num_workers=train_cfg.num_workers
        )
        valloader = data.DataLoader(
            val_set,
            batch_size=train_cfg.batch_size,
            sampler=CustomSamplerVal(val_list),
            num_workers=train_cfg.num_workers
        )

    # 设置模型
    if train_cfg.resume:
        if os.path.isfile(train_cfg.resume):
            logger.info(f"Loading checkpoint '{train_cfg.resume}'")
            model = torch.load(train_cfg.resume)
        else:
            logger.error(f"No checkpoint found at '{train_cfg.resume}'")
            return
    else:
        model = get_model(train_cfg.model_type, model_cfg.pretrained, model_cfg.num_classes)

    # 多GPU支持
    if train_cfg.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # 初始化评估指标跟踪器
    running_metrics = runningScore(model_cfg.num_classes)
    running_metrics_val = runningScore(model_cfg.num_classes)

    # 优化器设置 - 根据配置选择优化器
    if train_cfg.use_original_optim:
        logger.info("使用原始项目的优化器设置(Adam)")
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=optim_cfg.learning_rate,
                                   weight_decay=optim_cfg.weight_decay,
                                   amsgrad=True)
        scheduler = None  # 原始项目没有使用调度器
    elif train_cfg.model_type == 'i2u_net':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optim_cfg.learning_rate,
            weight_decay=optim_cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.n_epoch,
            eta_min=optim_cfg.min_lr
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
        scheduler = None

    # 损失函数设置
    loss_fn = core.loss.cross_entropy
    if train_cfg.class_weights:
        class_weights = torch.tensor(
            optim_cfg.class_weights,
            device=device,
            requires_grad=False,
            dtype=torch.float32  # 明确使用float32，保持一致性
        )
    else:
        class_weights = None

    best_iou = -100.0
    
    # 记录配置
    for key, value in vars(train_cfg).items():
        writer.add_text('Config/train', f'{key}: {value}', 0)
    for key, value in vars(model_cfg).items():
        writer.add_text('Config/model', f'{key}: {value}', 0)
    for key, value in vars(optim_cfg).items():
        writer.add_text('Config/optim', f'{key}: {value}', 0)
    
    # 记录模型初始化状态
    for name, param in model.named_parameters():
        writer.add_histogram(f'init/weight/{name}', param.data, 0)
        if param.grad is not None:
            writer.add_histogram(f'init/grad/{name}', param.grad.data, 0)

    # 添加训练模式信息
    if train_cfg.use_original_optim:
        logger.info("使用原始项目的优化器配置 (Adam, lr={})".format(optim_cfg.learning_rate))
    if train_cfg.use_original_sampler:
        logger.info("使用原始项目的采样器策略")
    if not train_cfg.use_amp:
        logger.info("禁用混合精度训练，与原始项目保持一致")
    else:
        logger.warning("启用混合精度训练，可能导致数值不稳定!")
    
    # 检查数据生成情况
    logger.info(f"训练集: {len(train_list)}个样本, 验证集: {len(val_list)}个样本")
    
    # 显示输入数据统计信息
    sample_img, _ = next(iter(trainloader))
    if not isinstance(sample_img, torch.Tensor):
        sample_img = sample_img[0] if isinstance(sample_img, list) else sample_img
    logger.info(f"输入数据形状: {sample_img.shape}, 范围: [{sample_img.min()}, {sample_img.max()}], 均值: {sample_img.mean()}")
    
    # 如果是I2U-Net，显示警告
    if train_cfg.model_type == 'i2u_net':
        logger.warning("使用I2U-Net模型，该模型比原始项目的deconvnet更复杂，可能需要不同的超参数!")
        logger.warning("如果训练不稳定或出现NaN，建议尝试设置--arch=section_deconvnet")

    # 训练循环
    for epoch in range(train_cfg.n_epoch):
        # 记录每个epoch的开始时间
        epoch_start_time = time.time()
        
        model.train()
        loss_train = 0.0
        total_iteration = 0
        
        # 临时禁用控制台日志，避免干扰进度条
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                handler._old_level = handler.level  # 保存原始级别
                handler.setLevel(logging.CRITICAL)  # 只显示严重错误
                
        # 修改使用tqdm的方式 - 使用更强的配置防止换行
        tbar = tqdm(
            range(len(trainloader)),
            desc=f'Epoch {epoch+1}/{train_cfg.n_epoch}',
            mininterval=5.0,     # 更长的最小更新间隔
            miniters=10,         # 更少的迭代更新
            dynamic_ncols=False, # 禁用动态列宽
            leave=True,          # 保留进度条
            position=0,          # 固定位置
            file=sys.stdout,     # 指定输出到标准输出
            ncols=120,           # 足够宽的固定列宽
            smoothing=0.1,       # 减少平滑，减少更新
            disable=False        # 确保不被禁用
        )
        
        # 在第一批次记录输入输出形状
        first_batch = True
        
        for i, (images, labels) in enumerate(trainloader):
            # 确保输入是4D的 [B, C, H, W]
            if images.dim() == 3:  # 如果是 [C, H, W]
                images = images.unsqueeze(0)  # 添加批次维度 [1, C, H, W]
            
            image_original, labels_original = images, labels
            images, labels = images.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            if train_cfg.use_amp:
                with autocast():
                    outputs = safe_forward(model, images, scaler)
                    if hasattr(model, 'get_attention') and train_cfg.log_attention:
                        attention_maps = model.get_attention()
            else:
                outputs = safe_forward(model, images, scaler)
                if hasattr(model, 'get_attention') and train_cfg.log_attention:
                    attention_maps = model.get_attention()
            
            # 打印第一批次的输入输出形状
            if first_batch:
                logger.info(f"Epoch {epoch+1} - 输入形状: {images.shape}, 输出形状: {outputs.shape}")
                first_batch = False
            
            probs = model.get_prob(outputs)
            
            # 打印输入和输出的形状
            if i == 0:
                print(f"Epoch {epoch} - 输入形状: {images.shape}, 输出形状: {outputs.shape}, 概率形状: {probs.shape}")
            
            # 计算损失
            if train_cfg.use_amp:
                with autocast():
                    loss = loss_fn(input=outputs, target=labels, weight=class_weights)
            else:
                loss = loss_fn(input=outputs, target=labels, weight=class_weights)
            
            # 检查NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"检测到NaN/Inf损失: {loss.item()}, 跳过此批次")
                # 打印输入统计信息以帮助调试
                logger.info(f"输入范围: [{images.min().item()}, {images.max().item()}], 均值: {images.mean().item()}")
                continue
            
            # 更新指标
            pred = outputs.detach().max(1)[1].cpu().numpy()
            gt = labels.detach().cpu().numpy()
            running_metrics.update(gt, pred)
            
            # 反向传播和优化
            if scaler is not None:
                scaler.scale(loss).backward()
                if train_cfg.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_cfg.clip_grad
                    )
                    
                    # 检查梯度是否出现NaN
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.warning(f"参数 {name} 的梯度出现NaN/Inf值")
                                # 使用与原始项目一致的策略：在出现NaN时不零化梯度
                                if not train_cfg.use_original_optim:
                                    param.grad.data.zero_()
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if train_cfg.clip_grad > 0 and not train_cfg.use_original_optim:
                    # 原始项目没有梯度裁剪，所以在use_original_optim模式下跳过
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_cfg.clip_grad
                    )
                    
                    # 检查梯度是否出现NaN
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.warning(f"参数 {name} 的梯度出现NaN/Inf值")
                                # 使用与原始项目一致的策略：在出现NaN时不零化梯度
                                if not train_cfg.use_original_optim:
                                    param.grad.data.zero_()
                
                optimizer.step()
            
            # 监控梯度统计(每隔一定步数)
            if i % 50 == 0:
                grad_stats, largest_grads = monitor_gradient_stats(model, writer, epoch, i)
                # 如果梯度异常大或异常小，输出警告
                if len(largest_grads) > 0 and largest_grads[0][1] > 10.0:
                    logger.warning(f"检测到大梯度: {largest_grads[0][0]} = {largest_grads[0][1]:.4f}")
            
            # 更新损失
            loss_train += loss.item()
            total_iteration += 1
            
            # 更新进度条，显示当前批次的损失
            tbar.set_postfix(loss=f"{loss.item():.3f}", avg_loss=f"{loss_train/total_iteration:.3f}")
            tbar.update(1)

            # 每隔log_freq个批次记录一次日志，不在进度条上重复显示
            if i % train_cfg.log_freq == 0:
                # 将迭代信息只记录到日志文件，不输出到控制台
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                        handler.setLevel(logging.WARNING)  # 临时调高控制台输出级别
                
                logger.info(
                    "Epoch [%d/%d] Iter [%d/%d] Loss: %.4f" %
                    (epoch + 1, train_cfg.n_epoch, i + 1, len(trainloader), loss.item())
                )
                
                # 恢复控制台日志级别
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                        handler.setLevel(logging.INFO)

            # 特征可视化
            if i == 0 and train_cfg.enable_visualization:
                try:
                    visualize_features(writer, model, images, epoch)
                except Exception as e:
                    logger.error(f"特征可视化失败: {e}")
                    # 如果失败，禁用后续可视化以避免重复错误
                    train_cfg.enable_visualization = False
                    logger.warning("已禁用特征可视化以避免后续错误")

            # 可视化注意力图（如果可用）
            if train_cfg.log_attention and not getattr(train_cfg, 'attention_visualization_error', False):
                try:
                    # 使用正确的方法名称
                    attention_map = model.get_attention()
                    if attention_map is not None:
                        input_size = (images.size(0), 1, images.size(2), images.size(3))
                        attention_map = visualize_attention(attention_map, input_size)
                        if attention_map is not None:
                            writer.add_image('attention_map', attention_map, epoch * len(trainloader) + i)
                except Exception as e:
                    # 只记录一次错误
                    logger.warning(f"注意力图可视化错误: {str(e)}")
                    # 设置标志，避免重复打印错误
                    train_cfg.attention_visualization_error = True
                    # 禁用后续注意力图可视化
                    train_cfg.log_attention = False

        # 确保循环结束后关闭进度条
        tbar.close()
        
        # 恢复控制台日志级别
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and hasattr(handler, '_old_level'):
                handler.setLevel(handler._old_level)
                delattr(handler, '_old_level')
        
        # 计算训练指标
        loss_train = loss_train / len(trainloader)
        score, class_iou = running_metrics.get_scores()
        
        # 记录训练指标
        writer.add_scalar('train/loss', loss_train, epoch + 1)
        
        # 记录所有指标，排除混淆矩阵
        for k, v in score.items():
            if k != 'confusion_matrix':
                writer.add_scalar(f'train/{k}', v, epoch + 1)
        
        # 可视化混淆矩阵
        if 'confusion_matrix' in score:
            cm = score['confusion_matrix']
            try:
                # 以图像形式记录混淆矩阵
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title("Train Confusion Matrix")
                plt.colorbar(im)
                
                # 转换为张量，然后记录
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = np.array(Image.open(buf))
                writer.add_image('train/confusion_matrix', img.transpose(2, 0, 1), epoch+1)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"记录训练混淆矩阵失败: {str(e)}")
        
        # 学习率调度器更新 - 仅在启用调度器时执行
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train/learning_rate', current_lr, epoch + 1)
            logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # 在epoch结束后计算花费的时间
        epoch_time = time.time() - epoch_start_time
        remaining_epochs = train_cfg.n_epoch - (epoch + 1)
        estimated_time_remaining = remaining_epochs * epoch_time / 3600.0  # 转换为小时
        
        # 记录时间信息
        time_metrics = timer.epoch_complete(epoch)
        writer.add_scalar('time/epoch_time', time_metrics['epoch_time'], epoch + 1)
        writer.add_scalar('time/estimated_remaining', time_metrics['estimated_remaining'], epoch + 1)
        
        # 记录模型参数统计
        if (epoch + 1) % 5 == 0:  # 每5轮记录一次
            for name, param in model.named_parameters():
                writer.add_histogram(f'params/{name}', param.data, epoch + 1)
                if param.grad is not None:
                    writer.add_histogram(f'grads/{name}', param.grad.data, epoch + 1)
        
        # 更新保存最佳模型的逻辑
        score, class_iou = running_metrics_val.get_scores()
        
        # 使用新的键名引用Mean IoU
        current_miou = score['Mean IoU']
        
        # 保存最佳模型
        if train_cfg.save_best and current_miou >= best_iou:
            best_iou = current_miou
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "best_iou": best_iou,
            }
            save_path = os.path.join(train_cfg.log_dir,
                                     f"best_model_miou_{best_iou:.4f}.pkl")
            torch.save(state, save_path)
            logger.info(f"Saved best model with IoU: {best_iou:.4f}")
        
        # 更新输出信息，使用修正后的时间估计和新的键名
        logger.info('Epoch %d completed. Avg loss: %.4f, Mean IoU: %.4f, Estimated time remaining: %.2fh' % (
            epoch+1, loss_train/total_iteration, 
            score['Mean IoU'], estimated_time_remaining))

        running_metrics.reset()

        # 验证循环
        with torch.no_grad():
            model.eval()
            loss_val = 0.0
            total_iteration_val = 0
            
            # 临时禁用控制台日志，避免干扰进度条
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    handler._old_level = handler.level  # 保存原始级别
                    handler.setLevel(logging.CRITICAL)  # 只显示严重错误
            
            # 添加验证进度条，与训练进度条配置一致
            tbar_val = tqdm(
                range(len(valloader)),
                desc=f'Validation {epoch+1}/{train_cfg.n_epoch}',
                mininterval=5.0,     # 更长的最小更新间隔
                miniters=8,          # 更少的迭代更新
                dynamic_ncols=False, # 禁用动态列宽
                leave=True,          # 保留进度条
                position=0,          # 固定位置
                file=sys.stdout,     # 指定输出到标准输出
                ncols=120,           # 足够宽的固定列宽
                smoothing=0.1,       # 减少平滑，减少更新
                disable=False        # 确保不被禁用
            )
            
            for i_val, (images_val, labels_val) in enumerate(valloader):
                # 确保输入是4D的 [B, C, H, W]
                if images_val.dim() == 3:  # 如果是 [C, H, W]
                    images_val = images_val.unsqueeze(0)  # 添加批次维度 [1, C, H, W]
                
                images_val = images_val.to(device)
                labels_val = labels_val.to(device)

                # 使用与训练一致的方式计算输出和损失
                if train_cfg.use_amp:
                    with autocast():
                        outputs = model(images_val)
                        loss = loss_fn(input=outputs, target=labels_val, weight=class_weights)
                else:
                    outputs = model(images_val)
                    loss = loss_fn(input=outputs, target=labels_val, weight=class_weights)
                
                pred = outputs.detach().max(1)[1].cpu().numpy()
                gt = labels_val.detach().cpu().numpy()
                running_metrics_val.update(gt, pred)
                loss_val += loss.item()
                total_iteration_val += 1
                
                # 更新验证进度条
                tbar_val.set_postfix(loss=f"{loss.item():.3f}")
                tbar_val.update(1)

            # 关闭验证进度条
            tbar_val.close()
            
            # 恢复控制台日志级别
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and hasattr(handler, '_old_level'):
                    handler.setLevel(handler._old_level)
                    delattr(handler, '_old_level')

            score, class_iou = running_metrics_val.get_scores()
            
            # 获取并输出完整的评估指标
            for k, v in score.items():
                if k != 'confusion_matrix':  # 不在控制台输出混淆矩阵
                    logger.info(f"{k}: {v:.4f}")
                    # 只记录标量值
                    writer.add_scalar(f'val/{k}', v, epoch+1)
                else:
                    # 对于混淆矩阵，可以使用image或figure方式记录
                    cm = score['confusion_matrix']
                    try:
                        # 以图像形式记录混淆矩阵
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                        ax.set_title("Confusion Matrix")
                        plt.colorbar(im)
                        
                        # 转换为张量，然后记录
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        img = np.array(Image.open(buf))
                        writer.add_image('val/confusion_matrix', img.transpose(2, 0, 1), epoch+1)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"记录混淆矩阵失败: {str(e)}")
            
            # 只在日志文件中记录混淆矩阵
            if 'confusion_matrix' in score:
                logger.debug(f"Confusion Matrix:\n{score['confusion_matrix']}")
            
            # 保存最佳模型
            if score['Mean IoU'] >= best_iou:
                best_iou = score['Mean IoU']
                model_path = os.path.join(log_dir, f"best_{train_cfg.model_type}.pth")
                if train_cfg.multi_gpu and hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model with IoU: {best_iou:.4f}")

            running_metrics_val.reset()

        # 定期保存检查点
        if (epoch + 1) % train_cfg.save_freq == 0:
            model_path = os.path.join(
                log_dir,
                f"{train_cfg.model_type}_epoch{epoch+1}.pth"
            )
            if train_cfg.multi_gpu and hasattr(model, 'module'):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

    writer.close()
    logger.info("Training completed!")


def get_arguments():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='i2u_net',
                        help='Network architecture. We have DeepLabv3_plus, DeepLabv2, FCN, PSPNet, etc.')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size. If None, automatically chosen based on GPU memory.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for optimizer')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--per_val', type=float, default=0.1,
                        help='Percentage of validation set')
    parser.add_argument('--per_val_f', type=float, default=0.1,
                        help='Percentage of validation set')
    parser.add_argument('--aug', action='store_true', default=True,
                        help='Whether to use data augmentation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--class_weights', action='store_true', default=True,
                        help='Whether to use class weights to reduce the effect of class imbalance')
    parser.add_argument('--no-visualization', action='store_true', default=False,
                        help='禁用特征可视化，可用于解决TensorBoard可视化问题')
    parser.add_argument('--disable-amp', action='store_true', default=True,
                        help='禁用混合精度训练，使用与原始项目相同的训练策略')
    parser.add_argument('--use-original-optim', action='store_true', default=True,
                        help='使用与原始项目相同的优化器设置(Adam)')
    parser.add_argument('--use-original-sampler', action='store_true', default=True,
                        help='使用与原始项目相同的采样器策略')
    parser.add_argument('--log-attention', action='store_true', default=False,
                        help='启用注意力图的记录和可视化（如果模型支持）')
    parser.add_argument('--no-save-best', action='store_true', default=False,
                        help='禁用保存最佳模型，只按照save_freq频率保存')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    train(args)
