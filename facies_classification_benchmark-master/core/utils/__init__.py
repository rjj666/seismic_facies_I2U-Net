"""
Utils package initialization
"""
import torch
import numpy as np

def np_to_tb(array):
    """
    将 numpy 数组转换为 TensorBoard 可视化格式
    
    Args:
        array (np.ndarray): 输入的 numpy 数组
        
    Returns:
        torch.Tensor: 转换后的张量，适用于 TensorBoard 可视化
    """
    if array.ndim == 2:
        array = array[np.newaxis, np.newaxis, ...]
    elif array.ndim == 3:
        array = array[np.newaxis, ...]
        
    array = np.transpose(array, (0, 3, 1, 2)) if array.shape[-1] == 3 else array
    
    return torch.from_numpy(array) 