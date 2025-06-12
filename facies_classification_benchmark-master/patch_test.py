import argparse
import os
from os.path import join as pjoin
import sys
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils import data
import torch.nn as nn

import torchvision.utils as vutils
from core.loader.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb
from core.models import get_model
from core.config import ModelConfig, OptimConfig, TrainingConfig
import logging
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 创建全局logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加控制台处理器
console_handler = logging.StreamHandler(sys.stderr)
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

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取模型参数配置
    train_cfg, model_cfg, _ = get_test_config(args)
    
    log_dir, model_name = os.path.split(args.model_path)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"测试模型: {args.model_path}")
    logger.info(f"测试分割: {args.split}")
    logger.info(f"使用内联: {args.inline}, 使用交叉线: {args.crossline}")
    
    # 创建模型架构
    model_type = args.model_type if args.model_type else train_cfg.model_type
    logger.info(f"使用模型类型: {model_type}")
    
    # 加载模型架构
    model = get_model(model_type, False, model_cfg.num_classes)
    
    # 加载保存的模型权重
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"成功加载模型权重")
    except Exception as e:
        logger.error(f"加载模型权重失败: {str(e)}")
        return
    
    model = model.to(device)  # 发送到可用GPU
    model.eval()  # 设置为评估模式
    
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(6)

    # 创建保存预测结果的目录
    pred_save_dir = os.path.join(log_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)
    logger.info(f"预测结果将保存到: {pred_save_dir}")

    # 如果是"both"，则测试两个区域，否则只测试指定区域
    if args.split == 'both':
        splits = ['test1', 'test2']
    else:
        splits = [args.split]
        
    logger.info(f"将在以下区域进行测试: {splits}")
    
    for sdx, split in enumerate(splits):
        # define indices of the array
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

        file_object = open(
            pjoin('data', 'splits', 'section_' + split + '.txt'), 'w')
        file_object.write('\n'.join(list_test))
        file_object.close()

        test_set = section_loader(is_transform=True,
                                  split=split,
                                  augmentations=None)
        n_classes = test_set.n_classes

        test_loader = data.DataLoader(test_set,
                                      batch_size=1,
                                      num_workers=4,
                                      shuffle=False)

        # print the results of this split:
        running_metrics_split = runningScore(n_classes)

        # 创建空数组用于存储预测结果
        # 使用与原始标签相同的形状
        predictions = np.zeros_like(labels)
        
        # 追踪当前处理的样本索引
        inline_idx = 0
        crossline_idx = 0

        # testing mode:
        with torch.no_grad():  # operations inside don't track history
            model.eval()
            total_iteration = 0
            for i, (images, labels) in enumerate(test_loader):
                total_iteration = total_iteration + 1
                image_original, labels_original = images, labels
                
                # 确保输入是4D的 [B, C, H, W]
                if images.dim() == 3:  # 如果是 [C, H, W]
                    images = images.unsqueeze(0)  # 添加批次维度 [1, C, H, W]
                    
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                pred = outputs.detach().max(1)[1].cpu().numpy()
                gt = labels.detach().cpu().numpy()
                
                # 为调试目的打印形状
                if i == 0:
                    logger.info(f"预测结果形状: {pred.shape}, 标签形状: {gt.shape}")
                
                running_metrics_split.update(gt, pred)
                running_metrics_overall.update(gt, pred)
                
                # 存储预测结果
                # 确定当前样本是inline还是crossline
                sample_name = list_test[i]
                try:
                    if sample_name.startswith('i_'):
                        # 这是inline样本
                        idx = int(sample_name[2:])
                        # 确保形状匹配 - 转置预测结果如果需要
                        if predictions[idx].shape != pred[0].shape:
                            logger.info(f"调整预测形状 - 从 {pred[0].shape} 到 {predictions[idx].shape}")
                            predictions[idx, :, :] = np.transpose(pred[0])
                        else:
                            predictions[idx, :, :] = pred[0]
                            
                    elif sample_name.startswith('x_'):
                        # 这是crossline样本
                        idx = int(sample_name[2:])
                        # 确保形状匹配 - 转置预测结果如果需要
                        if predictions[:, idx].shape != pred[0].shape:
                            logger.info(f"调整预测形状 - 从 {pred[0].shape} 到 {predictions[:, idx].shape}")
                            predictions[:, idx, :] = np.transpose(pred[0])
                        else:
                            predictions[:, idx, :] = pred[0]
                except Exception as e:
                    logger.error(f"错误: 在处理样本 {sample_name} 时出现问题: {str(e)}")
                    logger.error(f"预测形状: {pred[0].shape}, 目标形状(inline): {predictions[0].shape}, 目标形状(crossline): {predictions[:, 0].shape}")
                
                # 显示进度
                if i % 50 == 0 or i == len(list_test) - 1:
                    logger.info(f"处理 {split} 数据: {i+1}/{len(list_test)} 样本")

                numbers = [0, 99, 149, 399, 499]

                if i in numbers:
                    logger.info(f"保存可视化结果，样本索引: {i}")
                    
                    # 创建保存图像的目录
                    img_save_dir = os.path.join(log_dir, 'test_images')
                    os.makedirs(img_save_dir, exist_ok=True)
                    
                    # 修复原始图像可视化
                    original_img = image_original[0][0]
                    if original_img.dim() == 2:  # 如果是 [H, W]
                        original_img = original_img.unsqueeze(0)  # 添加通道维度 [1, H, W]
                    tb_original_image = vutils.make_grid(
                        original_img, normalize=True, scale_each=True)
                    writer.add_image(f'test/original_image_{i}',
                                     tb_original_image, 0, dataformats='CHW')
                    
                    # 保存为PNG文件
                    save_image(tb_original_image, os.path.join(img_save_dir, f'original_image_{i}.png'))

                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(np.squeeze(labels_original))
                    writer.add_image(f'test/original_label_{i}',
                                     np_to_tb(correct_label_decoded), 0, dataformats='NCHW')
                    
                    # 保存标签图像
                    plt.figure(figsize=(10, 8))
                    plt.imshow(correct_label_decoded)
                    plt.title(f"Original Label {i}")
                    plt.savefig(os.path.join(img_save_dir, f'original_label_{i}.png'))
                    plt.close()
                    
                    out = F.softmax(outputs, dim=1)

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    # 添加通道维度，确保是 [C, H, W] 格式
                    if confidence.dim() == 2:  # 如果是 [H, W]
                        confidence = confidence.unsqueeze(0)  # 添加通道维度 [1, H, W]
                    tb_confidence = vutils.make_grid(
                        confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(np.squeeze(prediction))
                    writer.add_image(f'test/predicted_{i}', np_to_tb(decoded), 0, dataformats='NCHW')
                    writer.add_image(f'test/confidence_{i}', tb_confidence, 0, dataformats='CHW')
                    
                    # 保存预测和置信度图像
                    plt.figure(figsize=(10, 8))
                    plt.imshow(decoded)
                    plt.title(f"Prediction {i}")
                    plt.savefig(os.path.join(img_save_dir, f'prediction_{i}.png'))
                    plt.close()
                    
                    save_image(tb_confidence, os.path.join(img_save_dir, f'confidence_{i}.png'))

                    # 可视化不同类别的热图
                    unary = outputs.cpu().detach()
                    unary_max = torch.max(unary)
                    unary_min = torch.min(unary)
                    unary = unary.add((-1*unary_min))
                    unary = unary/(unary_max - unary_min)

                    # 创建类别热图目录
                    class_heatmap_dir = os.path.join(img_save_dir, f'heatmaps_{i}')
                    os.makedirs(class_heatmap_dir, exist_ok=True)
                    
                    for channel in range(0, len(class_names)):
                        decoded_channel = unary[0][channel]
                        # 确保张量形状正确
                        if decoded_channel.dim() == 2:  # 如果是 [H, W]
                            decoded_channel = decoded_channel.unsqueeze(0)  # 添加通道维度 [1, H, W]
                        tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True)
                        writer.add_image(f'test_classes/_{class_names[channel]}_{i}', tb_channel, 0, dataformats='CHW')
                        
                        # 保存类别热图
                        save_image(tb_channel, os.path.join(class_heatmap_dir, f'{class_names[channel]}.png'))
                    
                    # 强制刷新，确保图像被写入
                    writer.flush()
                    
                    logger.info(f"已将样本 {i} 的可视化结果保存至 {img_save_dir}")

        # 保存预测结果为.npy文件
        pred_filename = os.path.join(pred_save_dir, f"{split}_predictions.npy")
        np.save(pred_filename, predictions)
        logger.info(f"已保存预测结果到: {pred_filename}")

        # get scores and save in writer()
        score, class_iou = running_metrics_split.get_scores()

        # 将分割结果添加到TensorBoard
        logger.info(f"区域 {split} 测试结果:")
        logger.info(f"Pixel Acc: {score['Pixel Acc']:.3f}")
        
        # Add split results to TB:
        writer.add_text(f'test__{split}/',
                        f'Pixel Acc: {score["Pixel Acc"]:.3f}', 0)
        for cdx, class_name in enumerate(class_names):
            # 使用 class_iou 字典获取每个类别的 IoU 值
            iou_value = class_iou[cdx] if cdx in class_iou else 0.0
            logger.info(f"  {class_name}_IoU {iou_value:.3f}")
            writer.add_text(
                f'test__{split}/', f'  {class_name}_IoU {iou_value:.3f}', 0)

        logger.info(f"Mean Class Acc: {score['Mean Class Acc']:.3f}")
        logger.info(f"Freq Weighted IoU: {score['Freq Weighted IoU']:.3f}")
        logger.info(f"Mean IoU: {score['Mean IoU']:.3f}")
        
        writer.add_text(
            f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc"]:.3f}', 0)
        writer.add_text(
            f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU"]:.3f}', 0)
        writer.add_text(f'test__{split}/',
                        f'Mean IoU: {score["Mean IoU"]:0.3f}', 0)

        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()

    # 记录最终测试结果
    logger.info('--------------- 最终测试结果 -----------------')
    logger.info(f'Pixel Acc: {score["Pixel Acc"]:.3f}')
    
    # Add split results to TB:
    writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc"]:.3f}', 0)
    for cdx, class_name in enumerate(class_names):
        # 使用 class_iou 字典获取每个类别的 IoU 值
        iou_value = class_iou[cdx] if cdx in class_iou else 0.0
        logger.info(f'     {class_name}_IoU {iou_value:.3f}')
        writer.add_text(
            'test_final', f'  {class_name}_IoU {iou_value:.3f}', 0)

    logger.info(f'Mean Class Acc: {score["Mean Class Acc"]:.3f}')
    logger.info(f'Freq Weighted IoU: {score["Freq Weighted IoU"]:.3f}')
    logger.info(f'Mean IoU: {score["Mean IoU"]:0.3f}')

    # 保存混淆矩阵
    confusion = score['confusion_matrix']
    np.savetxt(pjoin(log_dir,'confusion.csv'), confusion, delimiter=" ")
    logger.info(f"混淆矩阵已保存到 {pjoin(log_dir,'confusion.csv')}")

    # 确保所有数据都已写入
    writer.flush()
    writer.close()
    return

# 添加patch_label_2d函数
def patch_label_2d(model, img, patch_size, stride):
    img = torch.squeeze(img)
    h, w = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size/2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode='constant', value=0)

    num_classes = 6
    output_p = torch.zeros([1, num_classes, h+2*ps, w+2*ps])

    # generate output:
    for hdx in range(0, h-patch_size+ps, stride):
        for wdx in range(0, w-patch_size+ps, stride):
            patch = img_p[hdx + ps: hdx + ps + patch_size,
                          wdx + ps: wdx + ps + patch_size]
            patch = patch.unsqueeze(dim=0)  # channel dim
            patch = patch.unsqueeze(dim=0)  # batch dim

            assert (patch.shape == (1, 1, patch_size, patch_size))

            model_output = model(patch)
            output_p[:, :, hdx + ps: hdx + ps + patch_size, wdx + ps: wdx +
                     ps + patch_size] += torch.squeeze(model_output.detach().cpu())

    # crop the output_p in the middke
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output

def visualize_test_images(model_path, model_type='i2u_net', batch_size=1):
    """
    专门用于可视化测试图像的函数，使用原生PyTorch TensorBoard
    
    Args:
        model_path: 模型路径
        model_type: 模型类型
        batch_size: 批次大小
    """
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import os
    
    # 设置参数
    test_args = argparse.Namespace()
    test_args.model_path = model_path
    test_args.model_type = model_type
    test_args.split = 'both'
    test_args.crossline = True
    test_args.inline = True
    test_args.arch = model_type
    test_args.batch_size = batch_size
    test_args.train_patch_size = 256  # 添加训练patch大小
    test_args.test_stride = 128       # 添加测试步长
    
    # 设置日志目录
    log_dir = os.path.join(os.path.dirname(model_path), 'viz_images_' + 
                         datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取模型类型
    train_cfg = TrainingConfig()
    train_cfg.parse_config(os.path.join(os.path.dirname(model_path), 'train_config.yaml'))
    if test_args.model_type is None:
        test_args.model_type = train_cfg.arch
    
    # 创建输出目录
    logger.info(f"测试模型: {test_args.model_path}")
    logger.info(f"测试分割: {test_args.split}")
    logger.info(f"使用内联: {test_args.inline}, 使用交叉线: {test_args.crossline}")
    logger.info(f"使用模型类型: {test_args.model_type}")
    logger.info(f"使用patch大小: {test_args.train_patch_size}, 步长: {test_args.test_stride}")
    
    # 加载数据和模型
    model_config = ModelConfig()
    train_cfg.parse_config(os.path.join(os.path.dirname(model_path), 'train_config.yaml'))
    model_config.parse_config(os.path.join(os.path.dirname(model_path), 'model_config.yaml'))
    
    # 使用原生PyTorch的TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 处理不同的测试集
    for split_name in ['test1', 'test2']:
        if test_args.split == split_name or test_args.split == 'both':
            logger.info(f"处理测试区域: {split_name}")
            
            # 加载测试数据
            test_section = section_loader(split=split_name,
                                  is_transform=True,
                                  augmentations=None)
            logger.info(f"测试区域 {split_name} 形状: {test_section.seismic.shape}")
            logger.info(f"测试样本数量: {len(test_section)}")
            
            test_loader = data.DataLoader(test_section,
                                        batch_size=1,
                                        num_workers=4,
                                        shuffle=False)
            
            # 加载模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_file = test_args.model_path
            
            # 获取模型结构
            n_classes = model_config.n_classes
            if test_args.model_type == 'section_deconvnet':
                model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
                model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
            elif test_args.model_type == 'i2u_net':
                model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
                model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1), stride=(1, 1))
            
            # 加载权重
            state = torch.load(model_file, map_location=device)
            model.load_state_dict(state)
            model.to(device)
            logger.info("成功加载模型权重")
            
            # 可视化样本
            numbers = [0, 99, 149, 399, 499]
            n_classes = model_config.n_classes
            class_names = test_section.get_seismic_labels()
            
            logger.info(f"类别名称: {class_names}")
            model.eval()
            with torch.no_grad():
                for i, (images, labels) in enumerate(test_loader):
                    if i in numbers:
                        logger.info(f"处理样本 {split_name} - {i}")
                        image_original, labels_original = images, labels
                        
                        # 使用patch_label_2d进行预测
                        outputs = patch_label_2d(model=model,
                                               img=images,
                                               patch_size=test_args.train_patch_size,
                                               stride=test_args.test_stride)
                        
                        # 可视化原始图像
                        tb_original_image = vutils.make_grid(
                            image_original[0][0], normalize=True, scale_each=True)
                        writer.add_image(f'{split_name}/original_image_{i}',
                                       tb_original_image, 0)
                        
                        # 可视化标签
                        labels_original = labels_original.numpy()[0]
                        correct_label_decoded = test_section.decode_segmap(
                            np.squeeze(labels_original))
                        writer.add_image(f'{split_name}/original_label_{i}',
                                       np_to_tb(correct_label_decoded), 0, dataformats='NCHW')
                        
                        # 可视化预测
                        out = F.softmax(outputs, dim=1)
                        prediction = out.max(1)[1].cpu().numpy()[0]
                        confidence = out.max(1)[0].cpu().detach()[0]
                        tb_confidence = vutils.make_grid(
                            confidence, normalize=True, scale_each=True)
                        
                        decoded = test_section.decode_segmap(np.squeeze(prediction))
                        writer.add_image(f'{split_name}/predicted_{i}', np_to_tb(decoded), 0, dataformats='NCHW')
                        writer.add_image(f'{split_name}/confidence_{i}', tb_confidence, 0, dataformats='CHW')
                        
                        # 类别热图
                        for c in range(len(class_names)):
                            # 显示每个类别的预测概率
                            decoded_channel = unary[0][c]
                            tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True)
                            writer.add_image(f'{split_name}_classes/{class_names[c]}_{i}', tb_channel, 0, dataformats='CHW')
                        
                        writer.flush()
                    
                    if i > max(numbers):
                        break
    
    writer.close()
    logger.info(f"可视化完成，结果保存在 {log_dir}")
    
    # 返回日志目录用于查看
    return log_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='runs/Mar07_114540_i2u_net/best_i2u_net.pth',
                        help='Path to the saved model')
    parser.add_argument('--model_type', nargs='?', type=str, default=None,
                        help='模型类型，如i2u_net，section_deconvnet等')
    parser.add_argument('--split', nargs='?', type=str, default='both',
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    # 添加与训练参数对应的参数
    parser.add_argument('--arch', type=str, default='i2u_net',
                        help='Network architecture to use during testing')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--visualize_only', action='store_true',
                        help='只执行可视化，不进行完整测试')
    
    args = parser.parse_args()
    
    if args.visualize_only:
        visualize_test_images(args.model_path, args.model_type, args.batch_size)
    else:
        test(args)
