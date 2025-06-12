import torch
import torch.nn.functional as F

def cross_entropy(input, target, weight=None, ignore_index=255):
    '''
    Use 255 to fill empty values when padding or doing any augmentation operations
    like rotation. 
    '''
    # 确保目标是 2D 的 [B, H, W]
    target = torch.squeeze(target, dim=1)
    
    # 检查输入和目标的空间尺寸是否匹配
    if input.size(2) != target.size(1) or input.size(3) != target.size(2):
        # 调整输入以匹配目标的空间尺寸
        input = F.interpolate(input, size=(target.size(1), target.size(2)), 
                             mode='bilinear', align_corners=False)
    
    # 检查输入是否包含NaN或Inf
    if torch.isnan(input).any() or torch.isinf(input).any():
        # 替换NaN和Inf值
        input = torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 使用总和而不是平均值进行归约，以便更好地处理小批量
    loss = F.cross_entropy(input, target, weight, reduction='sum', ignore_index=255)
    # 计算有效像素数（不包括ignore_index）
    valid_pixels = (target != ignore_index).sum().item()
    if valid_pixels > 0:
        loss = loss / valid_pixels
    
    return loss
