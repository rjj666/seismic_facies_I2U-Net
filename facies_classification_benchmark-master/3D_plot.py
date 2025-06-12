###########  地震、标签、预测及叠加 四视图可视化    ################
import numpy as np
import cigvis
from matplotlib.colors import ListedColormap
import argparse
import os
import sys

# --- 辅助函数 ---

def create_label_colormap():
    """创建用于标签可视化的colormap"""
    colors = np.array([
        [69, 117, 180], [145, 191, 219], [224, 243, 248],
        [254, 224, 144], [252, 141, 89], [215, 48, 39]
    ]) / 255.0
    return ListedColormap(colors, name='LabelColorMap', N=len(colors))

def check_file_exists(path, name):
    """检查文件是否存在，如果不存在则打印错误并退出"""
    if not os.path.exists(path):
        print(f"错误: {name}文件不存在 - {path}")
        sys.exit(1)

# --- 可视化模式函数 ---

# 示例: python 3D_plot.py --mode three_view --split test1 --pred_path path/to/test1_predictions.npy
def plot_three_views(split, pred_path):
    """在1x3网格中显示地震、真实标签、预测标签"""
    print("模式: 三视图 (地震 | 真实标签 | 预测标签)")
    label_path = f'facies_classification_benchmark-master/data/test_once/{split}_labels.npy'
    seismic_path = f'facies_classification_benchmark-master/data/test_once/{split}_seismic.npy'

    check_file_exists(label_path, "真实标签")
    check_file_exists(seismic_path, "地震数据")
    check_file_exists(pred_path, "预测标签")

    labels = np.load(label_path)
    seismic = np.load(seismic_path)
    predictions = np.load(pred_path)

    print(f"加载数据: 标签={labels.shape}, 地震={seismic.shape}, 预测={predictions.shape}")

    if not (labels.shape == seismic.shape == predictions.shape):
        print(f"警告: 数据形状不匹配!")

    label_cmap = create_label_colormap()
    voxel_size = (1, 1, 0.3)

    # 地震节点
    vmax = np.percentile(np.abs(seismic), 98)
    seismic_nodes = cigvis.create_slices(seismic, cmap='gray', clim=[-vmax, vmax], voxel_size=voxel_size)

    # 真实标签节点
    label_nodes = cigvis.create_slices(labels, cmap=label_cmap, clim=[np.min(labels), np.max(labels)], voxel_size=voxel_size)

    # 预测标签节点
    pred_nodes = cigvis.create_slices(predictions, cmap=label_cmap, clim=[np.min(predictions), np.max(predictions)], voxel_size=voxel_size)

    all_plot_nodes = [seismic_nodes, label_nodes, pred_nodes]

    cigvis.plot3D(
        all_plot_nodes,
        grid=(1, 3),
        share=True,
        volume_rendering=False,
        slicer_kwargs={'interpolation': 'nearest'},
        bgcolor='white',
        size=(1800, 600) # 调整宽度以适应三个视图
    )

# 示例: python 3D_plot.py --mode single_seismic --data_path facies_classification_benchmark-master/data/test_once/test1_seismic.npy
def plot_single_seismic(data_path):
    """单独显示地震数据"""
    print("模式: 单视图 (地震数据)")
    check_file_exists(data_path, "地震数据")
    seismic = np.load(data_path)
    print(f"加载地震数据: {data_path}, 形状: {seismic.shape}")
    voxel_size = (1, 1, 0.3)
    vmax = np.percentile(np.abs(seismic), 98)
    seismic_nodes = cigvis.create_slices(seismic, cmap='gray', clim=[-vmax, vmax], voxel_size=voxel_size)
    cigvis.plot3D(seismic_nodes, slicer_kwargs={'interpolation': 'nearest'}, bgcolor='white', size=(800, 800))

# 示例 (真实标签): python 3D_plot.py --mode single_label --data_path facies_classification_benchmark-master/data/test_once/test1_labels.npy
# 示例 (预测标签): python 3D_plot.py --mode single_label --data_path path/to/test1_predictions.npy
def plot_single_label(data_path):
    """单独显示标签数据（真实或预测）"""
    print("模式: 单视图 (标签数据)")
    check_file_exists(data_path, "标签数据")
    labels = np.load(data_path)
    print(f"加载标签数据: {data_path}, 形状: {labels.shape}")
    label_cmap = create_label_colormap()
    voxel_size = (1, 1, 0.3)
    label_nodes = cigvis.create_slices(labels, cmap=label_cmap, clim=[np.min(labels), np.max(labels)], voxel_size=voxel_size)
    cigvis.plot3D(label_nodes, slicer_kwargs={'interpolation': 'nearest'}, bgcolor='white', size=(800, 800))

# 示例 (地震 vs 标签): python 3D_plot.py --mode compare_two --compare_mode seismic_label --split test1
# 示例 (标签 vs 预测): python 3D_plot.py --mode compare_two --compare_mode label_prediction --split test1 --pred_path path/to/test1_predictions.npy
def plot_compare_two(split, compare_mode, pred_path=None):
    """在1x2网格中比较两组数据"""
    print(f"模式: 双视图比较 ({compare_mode})")
    label_path = f'facies_classification_benchmark-master/data/test_once/{split}_labels.npy'
    seismic_path = f'facies_classification_benchmark-master/data/test_once/{split}_seismic.npy'

    check_file_exists(label_path, "真实标签")
    labels = np.load(label_path)
    label_cmap = create_label_colormap()
    voxel_size = (1, 1, 0.3)
    label_nodes = cigvis.create_slices(labels, cmap=label_cmap, clim=[np.min(labels), np.max(labels)], voxel_size=voxel_size)

    panel1_nodes = None
    panel2_nodes = None
    title = ""

    if compare_mode == 'seismic_label':
        check_file_exists(seismic_path, "地震数据")
        seismic = np.load(seismic_path)
        if labels.shape != seismic.shape:
             print(f"警告: 数据形状不匹配!")
        vmax = np.percentile(np.abs(seismic), 98)
        second_nodes = cigvis.create_slices(seismic, cmap='gray', clim=[-vmax, vmax], voxel_size=voxel_size)
        panel1_nodes = second_nodes # 地震放左边
        panel2_nodes = label_nodes   # 标签放右边
        title = f"{split}: 地震 vs 真实标签"

    elif compare_mode == 'label_prediction':
        if pred_path is None:
            print("错误: label_prediction 模式需要 --pred_path 参数")
            sys.exit(1)
        check_file_exists(pred_path, "预测标签")
        predictions = np.load(pred_path)
        if labels.shape != predictions.shape:
             print(f"警告: 数据形状不匹配!")
        second_nodes = cigvis.create_slices(predictions, cmap=label_cmap, clim=[np.min(predictions), np.max(predictions)], voxel_size=voxel_size)
        panel1_nodes = label_nodes       # 标签放左边
        panel2_nodes = second_nodes    # 预测放右边
        title = f"{split}: 真实标签 vs 预测标签"
    else:
        print(f"错误: 无效的比较模式 {compare_mode}")
        sys.exit(1)

    all_plot_nodes = [panel1_nodes, panel2_nodes]
    cigvis.plot3D(
        all_plot_nodes,
        grid=(1, 2),
        share=True,
        volume_rendering=False,
        slicer_kwargs={'interpolation': 'nearest'},
        bgcolor='white',
        size=(1200, 600),
        title=title
    )

# --- 命令行参数解析 --- 
def parse_args():
    parser = argparse.ArgumentParser(description='多功能3D可视化工具')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['three_view', 'single_seismic', 'single_label', 'compare_two'],
                        help='选择可视化模式')
    # --- three_view 和 compare_two (部分) 需要的参数 ---
    parser.add_argument('--split', type=str, default='test1', choices=['test1', 'test2'],
                        help='数据分割区域 (用于 three_view, compare_two)')
    # --- three_view 和 compare_two (label_prediction) 需要的参数 ---
    parser.add_argument('--pred_path', type=str, default=None,
                        help='预测标签 .npy 文件路径 (用于 three_view, compare_two[label_prediction])')
    # --- single_seismic 和 single_label 需要的参数 ---
    parser.add_argument('--data_path', type=str, default=None,
                        help='单个数据文件 .npy 路径 (用于 single_seismic, single_label)')
    # --- compare_two 需要的参数 ---
    parser.add_argument('--compare_mode', type=str, default='seismic_label',
                        choices=['seismic_label', 'label_prediction'],
                        help='比较模式 (用于 compare_two)')

    return parser.parse_args()

# --- 主逻辑 --- (确保在所有函数定义之后)
if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'three_view':
        if not args.pred_path:
            print("错误: three_view 模式需要 --pred_path 参数")
            sys.exit(1)
        plot_three_views(args.split, args.pred_path)

    elif args.mode == 'single_seismic':
        if not args.data_path:
            # 尝试从 split 推断默认路径
            args.data_path = f'facies_classification_benchmark-master/data/test_once/{args.split}_seismic.npy'
            print(f"提示: 未提供 --data_path，将使用默认地震路径: {args.data_path}")
        plot_single_seismic(args.data_path)

    elif args.mode == 'single_label':
        if not args.data_path:
             # 尝试从 split 推断默认真实标签路径
             args.data_path = f'facies_classification_benchmark-master/data/test_once/{args.split}_labels.npy'
             print(f"提示: 未提供 --data_path，将使用默认真实标签路径: {args.data_path}")
        plot_single_label(args.data_path)

    elif args.mode == 'compare_two':
        # 注意: plot_compare_two 内部会检查 pred_path 是否需要
        plot_compare_two(args.split, args.compare_mode, args.pred_path)

    else:
        print(f"错误: 未知的模式 '{args.mode}'")
        sys.exit(1)

'''
######################################      运行指令      ###################################
三视图              python facies_classification_benchmark-master/3D_plot.py --mode three_view --split test1 --pred_path path/to/test1_predictions.npy

单视图 (地震)        python facies_classification_benchmark-master/3D_plot.py --mode single_seismic --data_path facies_classification_benchmark-master/data/test_once/test1_seismic.npy
                   # 或让脚本尝试推断路径
                   python facies_classification_benchmark-master/3D_plot.py --mode single_seismic --split test1
                   
单视图（真实标签）     python facies_classification_benchmark-master/3D_plot.py --mode single_label --data_path facies_classification_benchmark-master/data/test_once/test1_labels.npy
                   # 或让脚本尝试推断路径
                   python facies_classification_benchmark-master/3D_plot.py --mode single_label --split test1
                
单视图 (预测标签)     python facies_classification_benchmark-master/3D_plot.py --mode single_label --data_path path/to/test1_predictions.npy

双视图 (地震 vs 标签) python facies_classification_benchmark-master/3D_plot.py --mode compare_two --compare_mode seismic_label --split test1

双视图 (标签 vs 预测) python facies_classification_benchmark-master/3D_plot.py --mode compare_two --compare_mode label_prediction --split test1 --pred_path path/to/test1_predictions.npy

'''


# ######地震+层位
# import numpy as np
# import cigvis
# from pathlib import Path
# root = Path(__file__).resolve().parent.parent.parent
#
# sxp = root / 'data/co2/sx.dat'
# sfp1 = root / 'data/co2/mh21.dat'
# sfp2 = root / 'data/co2/mh22.dat'
# ni, nx, nt = 192, 192, 240
#
# sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)
# sf1 = np.fromfile(sfp1, np.float32).reshape(ni, nx)
# sf2 = np.fromfile(sfp2, np.float32).reshape(ni, nx)
#
# nodes = cigvis.create_slices(sx)
#
# # show amplitude
# nodes += cigvis.create_surfaces([sf1, sf2],
#                                 volume=sx,
#                                 value_type='amp',
#                                 cmap='Petrel',
#                                 clim=[sx.min(), sx.max()])
#
# # add two points
# nodes += cigvis.create_points(np.array([[70, 50, 158], [20, 100, 80]]), r=3)
#
# cigvis.plot3D(nodes, size=(800, 800), savename='example.png')