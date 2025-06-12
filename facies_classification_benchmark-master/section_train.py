import argparse
import os
from datetime import datetime
from os.path import join as pjoin
from ast import arg
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

import core.loss
import torchvision.utils as vutils
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
    # Compose, RandomHorizontallyFlip, RandomRotate, AddNoise, PeruResize)
from core.loader.data_loader import *
from core.metrics import runningScore
from core.models import get_model
from core.utils import np_to_tb
import time
# Fix the random seeds:
torch.backends.cudnn.deterministic = True
torch.manual_seed(2019)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(2019)
np.random.seed(seed=2019)


def split_train_val(args, per_val=0.1):
    # create inline and crossline sections for training and validation:
    loader_type = 'section'
    labels = np.load(pjoin('data', 'train', 'train_labels.npy'))
    print(f"训练标签形状: {labels.shape}")  # 打印训练标签形状
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


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")  # 打印使用的设备
    
    # Generate the train and validation sets for the model:
    split_train_val(args, per_val=args.per_val)

    current_time = datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join('runs', current_time +
                           "_{}".format(args.arch))
    writer = SummaryWriter(log_dir=log_dir)
    # Setup Augmentations
    if args.aug:
        data_aug = Compose(
            [ RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
            # [PeruResize(), RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])       #####PeruResize方法用于Usegformer#################
    else:
        data_aug = None

    train_set = section_loader(is_transform=True,
                               split='train',
                               augmentations=data_aug)

    valdata_aug = Compose([])  # 修复：传入空列表作为augmentations参数
    # valdata_aug = Compose([PeruResize(),])                                                 ############PeruResize方法用于Usegformer#################
    val_set = section_loader(is_transform=True,split='val', augmentations=valdata_aug)

    n_classes = train_set.n_classes
    print(f"类别数量: {n_classes}")  # 打印类别数量

    # Create sampler:

    shuffle = False  # must turn False if using a custom sampler
    with open(pjoin('data', 'splits', 'section_train.txt'), 'r') as f:
        train_list = f.read().splitlines()
    with open(pjoin('data', 'splits', 'section_val.txt'), 'r') as f:
        val_list = f.read().splitlines()

    class CustomSamplerTrain(torch.utils.data.Sampler):
        def __iter__(self):
            char = ['i' if np.random.randint(2) == 1 else 'x']
            self.indices = [idx for (idx, name) in enumerate(
                train_list) if char[0] in name]
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

    class CustomSamplerVal(torch.utils.data.Sampler):
        def __iter__(self):
            char = ['i' if np.random.randint(2) == 1 else 'x']
            self.indices = [idx for (idx, name) in enumerate(
                val_list) if char[0] in name]
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

    trainloader = data.DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  sampler=CustomSamplerTrain(train_list),
                                  num_workers=8,
                                  shuffle=shuffle)
    valloader = data.DataLoader(val_set,
                                batch_size=args.batch_size,
                                sampler=CustomSamplerVal(val_list),
                                num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    else:
        model = get_model(args.arch, args.pretrained, n_classes, dim=args.model_dim)

    # Use as many GPUs as we can
    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)  # Send to GPU

    # PYTROCH NOTE: ALWAYS CONSTRUCT OPTIMIZERS AFTER MODEL IS PUSHED TO GPU/CPU,

    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        print('Using custom optimizer')
        optimizer = model.module.optimizer
    else:
        # optimizer = torch.optim.Adadelta(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)
        print('Using Adam optimizer with amsgrad=True')

    loss_fn = core.loss.cross_entropy
    # loss_fn = core.loss.improved_cross_entropy  # 使用新的 Lovasz-Softmax Loss

    if args.class_weights:
       # weights are inversely proportional to the frequency of the classes in the training set
        class_weights = torch.tensor(
            [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852], device=device, requires_grad=False)
        print(f"使用类权重: {class_weights}")  # 打印类权重
        # class_weights = None # 新加
    else:
        class_weights = None
        print("不使用类权重")

    best_iou = -100.0
    class_names = ['upper_ns', 'middle_ns', 'lower_ns',
                   'rijnland_chalk', 'scruff', 'zechstein']

    for arg in vars(args):
        text = arg + ': ' + str(getattr(args, arg))
        writer.add_text('Parameters/', text)

    # training
    for epoch in range(args.n_epoch):
        # Training Mode:
        model.train()
        loss_train, total_iteration = 0, 0

        for i, (images, labels) in enumerate(trainloader):
            if i == 0:  # 只在第一个批次打印形状信息
                print(f"训练数据形状 - 图像: {images.shape}, 标签: {labels.shape}")
                
            image_original, labels_original = images, labels
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)    ###########  segformer  #########

            if i == 0:  # 只在第一个批次打印形状信息
                print(f"模型输出形状: {outputs.shape}")
                
            pred = outputs.detach().max(1)[1].cpu().numpy()
            gt = labels.detach().cpu().numpy()
            running_metrics.update(gt, pred)

            loss = loss_fn(input=outputs, target=labels, weight=class_weights)
            loss_train += loss.item()
            loss.backward()

            # gradient clipping
            if args.clip != 0:
                # 修复deprecated警告
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_iteration = total_iteration + 1

            if (i) % 20 == 0:
                print("Epoch [%d/%d] training Loss: %.4f" %
                      (epoch + 1, args.n_epoch, loss.item()))

            numbers = [0]
            if i in numbers:
                # number 0 image in the batch
                tb_original_image = vutils.make_grid(
                    image_original[0][0], normalize=True, scale_each=True)
                writer.add_image('train/original_image',
                                 tb_original_image, epoch + 1)

                labels_original = labels_original.numpy()[0]
                correct_label_decoded = train_set.decode_segmap(
                    np.squeeze(labels_original))
                
                # 打印形状信息，帮助调试
                print(f"correct_label_decoded shape: {correct_label_decoded.shape}")
                
                # 打印np_to_tb输出的形状信息进行调试
                tb_label = np_to_tb(correct_label_decoded)
                print(f"np_to_tb输出形状: {tb_label.shape}, 类型: {type(tb_label)}")
                
                # 修复：添加dataformats='NCHW'参数
                writer.add_image('train/original_label',
                                tb_label, epoch + 1, dataformats='NCHW')
                
                out = F.softmax(outputs, dim=1)

                # this returns the max. channel number:
                prediction = out.max(1)[1].cpu().numpy()[0]
                # this returns the confidence:
                confidence = out.max(1)[0].cpu().detach()[0]
                tb_confidence = vutils.make_grid(
                    confidence, normalize=True, scale_each=True)

                decoded = train_set.decode_segmap(np.squeeze(prediction))
                
                # 打印形状信息，帮助调试
                print(f"decoded shape: {decoded.shape}")
                
                # 修复：添加dataformats='NCHW'参数
                tb_decoded = np_to_tb(decoded)
                print(f"decoded np_to_tb输出形状: {tb_decoded.shape}")
                writer.add_image('train/predicted', 
                                tb_decoded, epoch + 1, dataformats='NCHW')
                
                writer.add_image('train/confidence', tb_confidence, epoch + 1)

                unary = outputs.cpu().detach()
                unary_max = torch.max(unary)
                unary_min = torch.min(unary)
                unary = unary.add((-1*unary_min))
                unary = unary/(unary_max - unary_min)

                for channel in range(0, len(class_names)):
                    decoded_channel = unary[0][channel]
                    tb_channel = vutils.make_grid(
                        decoded_channel, normalize=True, scale_each=True)
                    writer.add_image(
                        f'train_classes/_{class_names[channel]}', tb_channel, epoch + 1)

        # Average metrics, and save in writer()
        loss_train /= total_iteration
        score, class_iou = running_metrics.get_scores()
        writer.add_scalar('train/Pixel Acc', score['Pixel Acc'], epoch+1)
        writer.add_scalar('train/Mean Class Acc',
                          score['Mean Class Acc'], epoch+1)
        writer.add_scalar('train/Freq Weighted IoU',
                          score['Freq Weighted IoU'], epoch+1)
        writer.add_scalar('train/Mean_IoU', score['Mean IoU'], epoch+1)
        running_metrics.reset()
        writer.add_scalar('train/loss', loss_train, epoch+1)

        if args.per_val != 0:
            with torch.no_grad():  # operations inside don't track history
                # Validation Mode:
                model.eval()
                loss_val, total_iteration_val = 0, 0

                for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                    if i_val == 0:  # 只在第一个批次打印形状信息
                        print(f"验证数据形状 - 图像: {images_val.shape}, 标签: {labels_val.shape}")
                        
                    image_original, labels_original = images_val, labels_val
                    images_val, labels_val = images_val.to(
                        device), labels_val.to(device)

                    outputs_val = model(images_val)
                    # outputs_val = F.interpolate(outputs_val, size=labels_val.shape[-2:], mode='bilinear', align_corners=False)    ############## segformer #########

                    if i_val == 0:  # 只在第一个批次打印形状信息
                        print(f"验证模型输出形状: {outputs_val.shape}")
                        
                    pred = outputs_val.detach().max(1)[1].cpu().numpy()
                    gt = labels_val.detach().cpu().numpy()

                    running_metrics_val.update(gt, pred)

                    loss = loss_fn(input=outputs_val, target=labels_val)
                    loss_val += loss.item()
                    total_iteration_val = total_iteration_val + 1

                    if (i_val) % 20 == 0:
                        print("Epoch [%d/%d] validation Loss: %.4f" %
                              (epoch, args.n_epoch, loss.item()))

                    numbers = [0]
                    if i_val in numbers:
                        # number 0 image in the batch
                        tb_original_image = vutils.make_grid(
                            image_original[0][0], normalize=True, scale_each=True)
                        writer.add_image('val/original_image',
                                         tb_original_image, epoch)
                        labels_original = labels_original.numpy()[0]
                        correct_label_decoded = train_set.decode_segmap(
                            np.squeeze(labels_original))
                            
                        # 打印形状信息，帮助调试
                        print(f"val correct_label_decoded shape: {correct_label_decoded.shape}")
                        
                        # 修复：添加dataformats='NCHW'参数
                        tb_val_label = np_to_tb(correct_label_decoded)
                        print(f"验证集np_to_tb输出形状: {tb_val_label.shape}")
                        writer.add_image('val/original_label',
                                        tb_val_label, epoch + 1, dataformats='NCHW')

                        out = F.softmax(outputs_val, dim=1)

                        # this returns the max. channel number:
                        prediction = out.max(1)[1].cpu().detach().numpy()[0]
                        # this returns the confidence:
                        confidence = out.max(1)[0].cpu().detach()[0]
                        tb_confidence = vutils.make_grid(
                            confidence, normalize=True, scale_each=True)

                        decoded = train_set.decode_segmap(np.squeeze(prediction))
                        
                        # 打印形状信息，帮助调试
                        print(f"val decoded shape: {decoded.shape}")
                        
                        # 修复：添加dataformats='NCHW'参数
                        tb_val_decoded = np_to_tb(decoded)
                        print(f"验证集decoded np_to_tb输出形状: {tb_val_decoded.shape}")
                        writer.add_image('val/predicted', 
                                        tb_val_decoded, epoch + 1, dataformats='NCHW')
                                            
                        writer.add_image('val/confidence',tb_confidence, epoch + 1)

                        unary = outputs_val.cpu().detach()  # 修复：使用outputs_val而不是outputs
                        unary_max, unary_min = torch.max(
                            unary), torch.min(unary)
                        unary = unary.add((-1*unary_min))
                        unary = unary/(unary_max - unary_min)

                        for channel in range(0, len(class_names)):
                            tb_channel = vutils.make_grid(
                                unary[0][channel], normalize=True, scale_each=True)
                            writer.add_image(
                                f'val_classes/_{class_names[channel]}', tb_channel, epoch + 1)

                # 计算平均验证损失
                loss_val /= total_iteration_val
                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)

                writer.add_scalar(
                    'val/Pixel Acc', score['Pixel Acc'], epoch+1)
                writer.add_scalar('val/Mean IoU', score['Mean IoU'], epoch+1)
                writer.add_scalar('val/Mean Class Acc',
                                  score['Mean Class Acc'], epoch+1)
                writer.add_scalar('val/Freq Weighted IoU',
                                  score['Freq Weighted IoU'], epoch+1)

                writer.add_scalar('val/loss', loss_val, epoch+1)  # 修复：使用平均验证损失
                running_metrics_val.reset()

                if score['Mean IoU'] >= best_iou:
                    best_iou = score['Mean IoU']
                    model_dir = os.path.join(
                        log_dir, f"{args.arch}_model.pkl")
                    torch.save(model, model_dir)
                    print(f"保存最佳模型，Mean IoU: {best_iou:.4f}")

        else:  # validation is turned off:
            # just save the latest model:
            if (epoch+1) % 10 == 0:
                model_dir = os.path.join(
                    log_dir, f"{args.arch}_ep{epoch+1}_model.pkl")
                torch.save(model, model_dir)
                print(f"保存第 {epoch+1} 轮模型")

    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='section_i2u_net',      ########修改为默认使用新模型############
                        help='Architecture to use [\'patch_deconvnet, path_deconvnet_skip, section_deconvnet, section_deconvnet_skip, i2u_net, section_i2u_net, TransUNet, UXlstmBot, segformer, usegformer, usegformerhyper\']')
    parser.add_argument('--model_dim', nargs='?', type=int, default=64,         ###########添加model_dim参数######
                        help='model_dim, use to segfomer and unet')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=61,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--clip', nargs='?', type=float, default=0.1,
                        help='Max norm of the gradients if clipping. Set to zero to disable. ')
    parser.add_argument('--per_val', nargs='?', type=float, default=0.1,
                        help='percentage of the training data for validation')
    parser.add_argument('--pretrained', nargs='?', type=bool, default=False,
                        help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument('--aug', nargs='?', type=bool, default=True,          ##############数据增强##################
                        help='Whether to use data augmentation.')
    parser.add_argument('--class_weights', nargs='?', type=bool, default=True,
                        help='Whether to use class weights to reduce the effect of class imbalance')


    args = parser.parse_args()
    print(args)
    train(args)
