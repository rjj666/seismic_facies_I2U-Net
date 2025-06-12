import argparse
from os.path import join as pjoin
import os  # 确保导入os模块

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torchvision.utils as vutils
from core.loader.data_loader import *
from core.metrics import runningScore
from core.utils import np_to_tb


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir, model_name = os.path.split(args.model_path)
    # load model:
    model = torch.load(args.model_path)
    model = model.to(device)  # Send to GPU if available
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']
    running_metrics_overall = runningScore(6)

    # 创建保存预测结果的目录
    pred_save_dir = os.path.join(log_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)
    print(f"预测结果将保存到: {pred_save_dir}")

    # 确保处理每个split时使用正确的数据
    splits = []
    if args.split == 'both':
        splits = ['test1', 'test2']
    else:
        splits = [args.split]
    
    print(f"将处理以下测试集: {splits}")

    for sdx, split in enumerate(splits):
        # define indices of the array
        labels = np.load(pjoin('data', 'test_once', split + '_labels.npy'))
        irange, xrange, depth = labels.shape
        print(f"加载 {split} 标签数据，形状: {labels.shape}")

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
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                pred = outputs.detach().max(1)[1].cpu().numpy()
                gt = labels.detach().cpu().numpy()
                
                # 为调试目的打印形状
                if i == 0:
                    print(f"预测结果形状: {pred.shape}, 标签形状: {gt.shape}")
                
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
                            print(f"调整预测形状 - 从 {pred[0].shape} 到 {predictions[idx].shape}")
                            predictions[idx, :, :] = np.transpose(pred[0])
                        else:
                            predictions[idx, :, :] = pred[0]
                            
                    elif sample_name.startswith('x_'):
                        # 这是crossline样本
                        idx = int(sample_name[2:])
                        # 确保形状匹配 - 转置预测结果如果需要
                        if predictions[:, idx].shape != pred[0].shape:
                            print(f"调整预测形状 - 从 {pred[0].shape} 到 {predictions[:, idx].shape}")
                            predictions[:, idx, :] = np.transpose(pred[0])
                        else:
                            predictions[:, idx, :] = pred[0]
                except Exception as e:
                    print(f"错误: 在处理样本 {sample_name} 时出现问题: {str(e)}")
                    print(f"预测形状: {pred[0].shape}, 目标形状(inline): {predictions[0].shape}, 目标形状(crossline): {predictions[:, 0].shape}")
                
                # 显示进度
                if i % 50 == 0 or i == len(list_test) - 1:
                    print(f"处理 {split} 数据: {i+1}/{len(list_test)} 样本")

                numbers = [0, 99, 149, 399, 499]  ##########记录的测试样本索引

                if i in numbers:
                    tb_original_image = vutils.make_grid(
                        image_original[0][0], normalize=True, scale_each=True)
                    writer.add_image('test/original_image',
                                     tb_original_image, i)

                    labels_original = labels_original.numpy()[0]
                    correct_label_decoded = test_set.decode_segmap(np.squeeze(labels_original))
                    
                    # 打印形状信息，帮助调试
                    print(f"correct_label_decoded shape: {correct_label_decoded.shape}")
                    
                    # 打印np_to_tb输出的形状信息进行调试
                    tb_label = np_to_tb(correct_label_decoded)
                    print(f"np_to_tb输出形状: {tb_label.shape}, 类型: {type(tb_label)}")
                    
                    writer.add_image('test/original_label',
                                     tb_label, i, dataformats='NCHW')
                    out = F.softmax(outputs, dim=1)

                    # this returns the max. channel number:
                    prediction = out.max(1)[1].cpu().numpy()[0]
                    # this returns the confidence:
                    confidence = out.max(1)[0].cpu().detach()[0]
                    tb_confidence = vutils.make_grid(
                        confidence, normalize=True, scale_each=True)

                    decoded = test_set.decode_segmap(np.squeeze(prediction))
                    
                    # 打印形状信息，帮助调试
                    print(f"decoded shape: {decoded.shape}")
                    
                    # 使用np_to_tb处理decoded，并打印形状
                    tb_decoded = np_to_tb(decoded)
                    print(f"decoded np_to_tb输出形状: {tb_decoded.shape}")
                    
                    writer.add_image('test/predicted', tb_decoded, i, dataformats='NCHW')
                    writer.add_image('test/confidence', tb_confidence, i)

                    # uncomment if you want to visualize the different class heatmaps
                    unary = outputs.cpu().detach()
                    unary_max = torch.max(unary)
                    unary_min = torch.min(unary)
                    unary = unary.add((-1*unary_min))
                    unary = unary/(unary_max - unary_min)

                    for channel in range(0, len(class_names)):
                        decoded_channel = unary[0][channel]
                        tb_channel = vutils.make_grid(decoded_channel, normalize=True, scale_each=True)
                        writer.add_image(f'test_classes/_{class_names[channel]}', tb_channel, i)

        # 保存预测结果为.npy文件
        pred_filename = os.path.join(pred_save_dir, f"{split}_predictions.npy")
        np.save(pred_filename, predictions)
        print(f"已保存预测结果到: {pred_filename}")
        print(f"预测结果形状: {predictions.shape}")

        # get scores and save in writer()
        score, class_iou = running_metrics_split.get_scores()
        
        # 打印score和class_iou的键以便调试
        print("==== SCORE KEYS ====")
        for key in score:
            print(f"Key: {key}, Type: {type(score[key])}")
        print("==== CLASS_IOU KEYS ====")
        print(f"class_iou type: {type(class_iou)}, shape: {np.shape(class_iou)}")
        
        # Add split results to TB:
        writer.add_text(f'test__{split}/',
                        f'Pixel Acc: {score["Pixel Acc"]:.3f}', 0)
        for cdx, class_name in enumerate(class_names):
            writer.add_text(
                f'test__{split}/', f'  {class_name}_accuracy {class_iou[cdx]:.3f}', 0)

        writer.add_text(
            f'test__{split}/', f'Mean Class Acc: {score["Mean Class Acc"]:.3f}', 0)
        writer.add_text(
            f'test__{split}/', f'Freq Weighted IoU: {score["Freq Weighted IoU"]:.3f}', 0)
        writer.add_text(f'test__{split}/',
                        f'Mean IoU: {score["Mean IoU"]:0.3f}', 0)

        running_metrics_split.reset()

    # FINAL TEST RESULTS:
    score, class_iou = running_metrics_overall.get_scores()
    
    # 打印最终score和class_iou的信息
    print("==== FINAL SCORE KEYS ====")
    for key in score:
        print(f"Key: {key}, Type: {type(score[key])}")
    print("==== FINAL CLASS_IOU KEYS ====")
    print(f"class_iou type: {type(class_iou)}, shape: {np.shape(class_iou)}")

    # Add split results to TB:
    writer.add_text('test_final', f'Pixel Acc: {score["Pixel Acc"]:.3f}', 0)
    for cdx, class_name in enumerate(class_names):
        writer.add_text(
            'test_final', f'  {class_name}_accuracy {class_iou[cdx]:.3f}', 0)

    writer.add_text(
        'test_final', f'Mean Class Acc: {score["Mean Class Acc"]:.3f}', 0)
    writer.add_text(
        'test_final', f'Freq Weighted IoU: {score["Freq Weighted IoU"]:.3f}', 0)
    writer.add_text('test_final', f'Mean IoU: {score["Mean IoU"]:0.3f}', 0)

    print('--------------- FINAL RESULTS -----------------')
    print(f'Pixel Acc: {score["Pixel Acc"]:.3f}')
    for cdx, class_name in enumerate(class_names):
        print(
            f'     {class_name}_accuracy {class_iou[cdx]:.3f}')
    print(f'Mean Class Acc: {score["Mean Class Acc"]:.3f}')
    print(f'Freq Weighted IoU: {score["Freq Weighted IoU"]:.3f}')
    print(f'Mean IoU: {score["Mean IoU"]:0.3f}')

    # Save confusion matrix:
    confusion = score['confusion_matrix']
    np.savetxt(pjoin(log_dir,'confusion.csv'), confusion, delimiter=" ")

    writer.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='runs/Mar14_162725_section_i2u_net/section_i2u_net_model.pkl',
                        help='Path to the saved model')
    parser.add_argument('--split', nargs='?', type=str, default='both',                            #指定要在'test1'、'test2'或两者（'both'）上进行测试
                        help='Choose from: "test1", "test2", or "both" to change which region to test on')
    parser.add_argument('--crossline', nargs='?', type=bool, default=True,
                        help='whether to test in crossline mode')
    parser.add_argument('--inline', nargs='?', type=bool, default=True,
                        help='whether to test inline mode')
    args = parser.parse_args()
    test(args)

