import argparse
import os
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torchvision.utils as vutils
from core.loader.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb
from core.models import get_model
from core.config import ModelConfig, OptimConfig, TrainingConfig
import logging
import sys

# 创建logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 自定义的简化版get_config函数，适用于测试脚本
def get_test_config(args):
    """
    为测试创建配置对象
    """
    train_cfg = TrainingConfig()
    
    # 从args更新model_type
    if args.model_type:
        train_cfg.model_type = args.model_type
    elif args.arch:
        train_cfg.model_type = args.arch
        
    model_cfg = ModelConfig()
    optim_cfg = OptimConfig()
    
    return train_cfg, model_cfg, optim_cfg

def save_test_results(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取模型参数配置
    train_cfg, model_cfg, _ = get_test_config(args)
    
    log_dir, model_name = os.path.split(args.model_path)
    output_dir = args.output_dir if args.output_dir else os.path.join(log_dir, 'npz_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(output_dir, 'save_results.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"测试区域: {args.split}")
    logger.info(f"使用内联: {args.inline}, 使用交叉线: {args.crossline}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建模型架构
    model_type = args.model_type if args.model_type else train_cfg.model_type
    logger.info(f"使用模型类型: {model_type}")
    
    # 加载模型
    try:
        if os.path.splitext(args.model_path)[1] == '.pth':
            # 加载模型架构
            model = get_model(model_type, False, model_cfg.num_classes)
            # 加载权重
            state_dict = torch.load(args.model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            # 直接加载完整模型
            model = torch.load(args.model_path, map_location=device)
        
        model = model.to(device)  # 发送到可用GPU
        model.eval()  # 设置为评估模式
        logger.info(f"成功加载模型")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return
    
    # 指定要保存的样本索引
    numbers = [0, 99, 149, 399, 499]
    
    # 如果是"both"，则测试两个区域，否则只测试指定区域
    if args.split == 'both':
        splits = ['test1', 'test2']
    else:
        splits = [args.split]
    
    logger.info(f"将处理以下区域: {splits}")
    logger.info(f"将保存以下索引的样本: {numbers}")
    
    for split in splits:
        # 定义标签数组形状
        labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
        irange, xrange, depth = labels.shape
        logger.info(f"测试区域 {split} 形状: {labels.shape}")
        
        if args.inline:
            i_list = list(range(irange))
            i_list = ['i_'+str(inline) for inline in i_list]
        else:
            i_list = []
            
        if args.crossline:
            x_list = list(range(xrange))
            x_list = ['x_'+str(crossline) for crossline in x_list]
        else:
            x_list = []
            
        list_test = i_list + x_list
        logger.info(f"测试样本数量: {len(list_test)}")
        
        # 写入测试样本列表文件
        file_object = open(
            pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
        file_object.write('\n'.join(list_test))
        file_object.close()
        
        # 加载测试数据
        test_set = section_loader(is_transform=True,
                                 split=split,
                                 augmentations=None)
        
        test_loader = data.DataLoader(test_set,
                                     batch_size=1,
                                     num_workers=4,
                                     shuffle=False)
        
        # 以无梯度模式进行预测
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if i in numbers:
                    logger.info(f"处理样本 {split}_{i}")
                    
                    # 获取原始图像和标签
                    image_original, labels_original = images.clone(), labels.clone()
                    
                    # 将数据发送到设备
                    images, labels = images.to(device), labels.to(device)
                    
                    # 模型推理
                    outputs = model(images)
                    
                    # 获取预测结果
                    out = F.softmax(outputs, dim=1)
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    
                    # 保存为npz文件
                    # 原始图像是单通道的，取第一个批次、第一个通道
                    original_image = image_original[0][0].cpu().numpy()
                    
                    # 原始标签同样是单通道
                    original_label = labels_original[0].cpu().numpy()
                    
                    # 保存数据
                    save_path = os.path.join(output_dir, f"{split}_{i}_res.npz")
                    np.savez(save_path,
                             origin=original_image.astype(np.float32),
                             originlabel=original_label.astype(np.uint8),
                             pred=prediction.astype(np.uint8))
                    
                    logger.info(f"已保存结果到 {save_path}")
                    
                    # 打印一些形状信息，用于调试
                    logger.info(f"原始图像形状: {original_image.shape}, 类型: {original_image.dtype}")
                    logger.info(f"原始标签形状: {original_label.shape}, 类型: {original_label.dtype}")
                    logger.info(f"预测结果形状: {prediction.shape}, 类型: {prediction.dtype}")
                
                # 如果已经处理完所有需要保存的样本，可以提前退出循环
                if i > max(numbers):
                    break
                    
        logger.info(f"完成区域 {split} 的处理")
    
    logger.info("所有样本处理完毕")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='保存测试结果为npz文件')
    parser.add_argument('--model_path', nargs='?', type=str, default='runs/Mar14_162725_section_i2u_net/section_i2u_net_model.pkl',
                        help='模型路径')
    parser.add_argument('--model_type', nargs='?', type=str, default=None,
                        help='模型类型，如i2u_net，section_deconvnet等')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='选择测试区域: "test1", "test2", 或 "both"')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='是否测试交叉线模式')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='是否测试内联模式')
    parser.add_argument('--output_dir', nargs='?', type=str, default=None,
                        help='输出目录，默认为模型目录下的npz_results子目录')
    parser.add_argument('--arch', type=str, default='i2u_net',
                        help='网络架构类型')
    
    args = parser.parse_args()
    save_test_results(args) 