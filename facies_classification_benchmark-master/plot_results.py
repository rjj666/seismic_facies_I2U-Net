#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化测试结果
根据save_test_results.py保存的.npz文件生成可视化图像
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import glob

# 设置Seaborn风格
sns.set_style('whitegrid')

# 类别颜色定义
LABEL_COLORS = np.asarray([
    [69, 117, 180],  # 上NS - 蓝色
    [145, 191, 219],  # 中NS - 浅蓝色
    [224, 243, 248],  # 下NS - 淡蓝色
    [254, 224, 144],  # Rijnland和白垩 - 淡黄色
    [252, 141, 89],  # Scruff - 橙色
    [215, 48, 39]  # Zechstein - 红色
])

# 类别名称
CLASS_NAMES = ['上NS', '中NS', '下NS', 'Rijnland和白垩', 'Scruff', 'Zechstein']

# 默认参数设置
DEFAULT_INPUT_DIR = 'runs'  # 默认查找最新的输出目录
DEFAULT_DPI = 600
DEFAULT_SAVE_FORMAT = 'png'
DEFAULT_ALPHA = 0.7


def find_latest_npz_dir():
    """查找最新的包含.npz文件的目录"""
    if not os.path.exists(DEFAULT_INPUT_DIR):
        return None
        
    # 查找runs目录下的所有子目录
    model_dirs = [d for d in os.listdir(DEFAULT_INPUT_DIR) 
                 if os.path.isdir(os.path.join(DEFAULT_INPUT_DIR, d))]
    
    # 按修改时间排序
    model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(DEFAULT_INPUT_DIR, x)), reverse=True)
    
    # 检查每个目录是否包含npz_results子目录
    for model_dir in model_dirs:
        npz_dir = os.path.join(DEFAULT_INPUT_DIR, model_dir, 'npz_results')
        if os.path.exists(npz_dir) and glob.glob(os.path.join(npz_dir, '*_res.npz')):
            return npz_dir
            
    # 如果没有找到npz_results目录，检查是否直接包含.npz文件
    for model_dir in model_dirs:
        full_dir = os.path.join(DEFAULT_INPUT_DIR, model_dir)
        if glob.glob(os.path.join(full_dir, '*_res.npz')):
            return full_dir
            
    return None


def decode_segmap(label_mask, plot=False):
    """将分割标签解码成彩色图像

    Args:
        label_mask (np.ndarray): 包含整数类别标签的(M,N)数组
        plot (bool): 是否在图中显示彩色图像

    Returns:
        np.ndarray: 解码后的彩色图像
    """
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    for ll in range(0, 6):
        r[label_mask == ll] = LABEL_COLORS[ll, 0]
        g[label_mask == ll] = LABEL_COLORS[ll, 1]
        b[label_mask == ll] = LABEL_COLORS[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def reverse_segmap(segmap):
    """从RGB颜色映射反向提取类别标签

    Args:
        segmap (np.ndarray): RGB格式的分割图

    Returns:
        np.ndarray: 类别索引图
    """
    result = np.zeros(segmap.shape[:2], dtype=np.uint8)

    for ll in range(0, 6):
        mask = np.all(segmap == LABEL_COLORS[ll], axis=2)
        result[mask] = ll

    return result


def create_label_colormap():
    """创建用于分割可视化的颜色映射

    Returns:
        ListedColormap: matplotlib颜色映射对象
    """
    return ListedColormap(LABEL_COLORS / 255.0)


def ensure_2d(array):
    """确保数组是2D的，无论输入维度如何
    
    Args:
        array (np.ndarray): 输入数组
        
    Returns:
        np.ndarray: 2D数组
    """
    # 如果是3D或更高维度，取第一个通道或批次
    if array.ndim > 2:
        # 如果第一个维度是batch维度（通常为1），取第一个batch
        if array.shape[0] == 1:
            return array[0] if array[0].ndim == 2 else array[0, 0]
        # 如果是通道在前格式 (C, H, W)，取第一个通道
        elif array.shape[0] <= 3:  # 假设最多3个通道（RGB）
            return array[0]
        # 其他情况可能是 (H, W, C) 格式
        else:
            return array[:, :, 0] if array.ndim == 3 else array
    return array


def create_overlay_manually(seismic_img, label_img, alpha=0.7):
    """手动创建标签覆盖图，解决类别0不显示的问题
    
    Args:
        seismic_img (np.ndarray): 地震数据图像，灰度uint8格式
        label_img (np.ndarray): 标签图像，uint8格式(0-5)
        alpha (float): 透明度
        
    Returns:
        np.ndarray: 覆盖后的图像
    """
    # 创建RGB版本的地震图
    if seismic_img.ndim == 2:
        seismic_rgb = np.stack([seismic_img] * 3, axis=2)
    else:
        seismic_rgb = seismic_img
        
    # 确保类型为uint8
    seismic_rgb = seismic_rgb.astype(np.uint8)
    
    # 创建彩色标签图
    label_colored = np.zeros((*label_img.shape, 3), dtype=np.uint8)
    
    # 记录哪些位置有标签
    label_mask = np.zeros(label_img.shape, dtype=bool)
    
    # 为每个类别赋予颜色
    for cls_id in range(6):
        mask = (label_img == cls_id)
        if np.any(mask):  # 如果有这个类别的像素
            label_mask = label_mask | mask
            for c in range(3):
                label_colored[mask, c] = LABEL_COLORS[cls_id, c]
    
    # 使用alpha混合创建最终图像
    result = np.zeros_like(seismic_rgb)
    for c in range(3):
        result[:, :, c] = np.where(
            label_mask,
            alpha * label_colored[:, :, c] + (1 - alpha) * seismic_rgb[:, :, c],
            seismic_rgb[:, :, c]
        )
    
    return result


def plot_results(npz_file, output_dir=None, dpi=DEFAULT_DPI, alpha=DEFAULT_ALPHA, save_format=DEFAULT_SAVE_FORMAT):
    """从.npz文件可视化测试结果

    Args:
        npz_file (str): .npz文件路径
        output_dir (str): 输出目录，默认为None表示与输入文件同目录
        dpi (int): 输出图像DPI
        alpha (float): 分割图覆盖在原图上的透明度
        save_format (str): 保存格式，'png'或'pdf'
    """
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(npz_file)
    os.makedirs(output_dir, exist_ok=True)

    # 获取基础文件名，用于构建输出文件名
    base_name = os.path.basename(npz_file).replace('_res.npz', '')

    # 加载.npz文件
    try:
        data = np.load(npz_file)
        seismic = data["origin"]  # 原始地震数据
        label = data["originlabel"]  # 真实标签
        pred = data["pred"]  # 预测结果

        print(f"加载文件: {npz_file}")
        print(f"  地震数据形状: {seismic.shape}, 类型: {seismic.dtype}")
        print(f"  标签形状: {label.shape}, 类型: {label.dtype}")
        print(f"  预测形状: {pred.shape}, 类型: {pred.dtype}")
        
        # 确保数据是2D的
        seismic_2d = ensure_2d(seismic)
        label_2d = ensure_2d(label)
        pred_2d = ensure_2d(pred)
        
        print(f"  处理后地震数据形状: {seismic_2d.shape}")
        print(f"  处理后标签形状: {label_2d.shape}")
        print(f"  处理后预测形状: {pred_2d.shape}")
        
        # 检查并打印每个类别的像素数量，帮助调试
        for cls_id in range(6):
            cls_pixels_in_label = np.sum(label_2d == cls_id)
            cls_pixels_in_pred = np.sum(pred_2d == cls_id)
            if cls_pixels_in_label > 0 or cls_pixels_in_pred > 0:
                print(f"  类别 {cls_id} ({CLASS_NAMES[cls_id]}): "
                      f"标签中 {cls_pixels_in_label} 像素, "
                      f"预测中 {cls_pixels_in_pred} 像素")
        
    except Exception as e:
        print(f"无法加载{npz_file}: {str(e)}")
        return

    # 创建图像
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['figure.dpi'] = dpi

    # 1. 原始地震图
    plt.figure(figsize=(10, 8))
    plt.imshow(seismic_2d, cmap="gray")
    plt.title("原始地震数据", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_seismic.{save_format}"))
    plt.close()

    # 2. 真实标签
    plt.figure(figsize=(10, 8))
    label_rgb = decode_segmap(label_2d)
    plt.imshow(label_rgb)
    plt.title("真实标签", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_true_label.{save_format}"))
    plt.close()

    # 3. 预测结果
    plt.figure(figsize=(10, 8))
    pred_rgb = decode_segmap(pred_2d)
    plt.imshow(pred_rgb)
    plt.title("预测结果", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_prediction.{save_format}"))
    plt.close()

    # 4. 真实标签覆盖在地震数据上 - 使用新方法
    plt.figure(figsize=(10, 8))
    
    # 归一化到0-255用于可视化
    seismic_uint8 = ((seismic_2d - seismic_2d.min()) /
                     (seismic_2d.max() - seismic_2d.min()) * 255).astype(np.uint8)
    
    # 使用自定义函数创建覆盖图，确保所有类别（包括0类别）都能正确显示
    overlay_true = create_overlay_manually(seismic_uint8, label_2d.astype(np.uint8), alpha)
    
    plt.imshow(overlay_true)
    plt.title("真实标签覆盖", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_true_overlay.{save_format}"))
    plt.close()

    # 5. 预测结果覆盖在地震数据上 - 使用新方法
    plt.figure(figsize=(10, 8))
    
    # 使用自定义函数创建覆盖图
    overlay_pred = create_overlay_manually(seismic_uint8, pred_2d.astype(np.uint8), alpha)
    
    plt.imshow(overlay_pred)
    plt.title("预测结果覆盖", fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_pred_overlay.{save_format}"))
    plt.close()

    # 6. 类别图例
    plt.figure(figsize=(10, 2))
    legend_img = np.zeros((100, 600, 3))
    width = 600 // 6

    for i in range(6):
        legend_img[:, i * width:(i + 1) * width, :] = LABEL_COLORS[i] / 255.0

    plt.imshow(legend_img)
    plt.xticks([width // 2 + i * width for i in range(6)], CLASS_NAMES, fontsize=12)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_legend.{save_format}"))
    plt.close()

    print(f"所有图像已保存到 {output_dir}")


def process_directory(input_dir, output_dir=None, pattern='*_res.npz', dpi=DEFAULT_DPI, alpha=DEFAULT_ALPHA, save_format=DEFAULT_SAVE_FORMAT):
    """处理目录中的所有.npz文件

    Args:
        input_dir (str): 输入目录
        output_dir (str): 输出目录
        pattern (str): 文件匹配模式
        dpi (int): 图像DPI
        alpha (float): 透明度
        save_format (str): 保存格式
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'plots')

    os.makedirs(output_dir, exist_ok=True)

    npz_files = glob.glob(os.path.join(input_dir, pattern))

    if not npz_files:
        print(f"在 {input_dir} 中没有找到匹配 {pattern} 的文件")
        return

    print(f"找到 {len(npz_files)} 个文件，开始处理...")

    for npz_file in npz_files:
        plot_results(npz_file, output_dir, dpi, alpha, save_format)

    print("所有文件处理完成")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化地震数据分割结果')

    parser.add_argument('--input', type=str, default=None,
                        help='输入.npz文件路径或包含.npz文件的目录，默认自动查找最新目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录，默认为输入目录或输入目录下的plots子目录')
    parser.add_argument('--pattern', type=str, default='*_res.npz',
                        help='当输入为目录时的文件匹配模式')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                        help='输出图像DPI')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                        help='分割图覆盖原图的透明度')
    parser.add_argument('--format', type=str, default=DEFAULT_SAVE_FORMAT, choices=['png', 'pdf', 'jpg'],
                        help='保存图像格式')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 如果没有指定输入目录，尝试自动查找最新的包含.npz文件的目录
    input_path = args.input
    if input_path is None:
        input_path = find_latest_npz_dir()
        if input_path:
            print(f"自动选择最新的.npz文件目录: {input_path}")
        else:
            print("未找到包含.npz文件的目录，请手动指定--input参数")
            return

    # 检查输入是文件还是目录
    if os.path.isfile(input_path):
        plot_results(input_path, args.output, args.dpi, args.alpha, args.format)
    elif os.path.isdir(input_path):
        process_directory(input_path, args.output, args.pattern, args.dpi, args.alpha, args.format)
    else:
        print(f"错误: 输入 {input_path} 既不是文件也不是目录")


if __name__ == '__main__':
    main() 
