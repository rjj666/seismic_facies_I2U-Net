# 变更日志 (Changelog)

所有显著变更都将记录在此文件中。

## [未发布]

### 添加内容
- 创建I2U-Net网络架构的drawio文件 (`i2u_net_architecture.drawio`)，包含网络整体结构和各模块详细说明
- 创建Section I2U-Net网络架构的drawio文件 (`section_i2u_net_architecture.drawio`)，展示改进版本的详细结构
- 创建ECA模块(高效通道注意力机制)的详细结构图 (`eca_module.drawio`)
- 创建多特征交互集成解码器块的详细结构图 (`mfii_decoder_block.drawio`)
- 创建空间金字塔池化Inception模块的详细结构图 (`spp_inception_block.drawio`)

### 特性说明
- 所有drawio文件均严格遵循plot_rules中规定的绘图规范
- 添加了模块间的详细连接关系和数据流向
- 提供了每个模块的功能详细说明
- 特别强调了Section I2U-Net相比原始I2U-Net的改进之处
- 所有图示均包含足够的细节，便于理解网络架构

### 添加
- 新增 `Section_I2U_Net` 模型，支持任意尺寸输入，特别适用于Inline主测线（701×255）和Xline联络测线（401×255）整个section大小的数据
- 在 `SPP_inception_block` 中使用自适应池化层，以适应不同尺寸的输入
- 在模型前向传播中添加输入尺寸检查，确保输出尺寸与输入匹配
- 在 `section_train.py` 中添加 `model_dim` 参数，用于配置模型维度
- 在 `section_train.py` 中添加更多调试信息打印，包括数据形状、设备信息等
- 在 `section_test.py` 中添加调试信息输出，用于监控图像处理和TensorBoard记录过程
- 全面重构 `plot.py` 脚本，添加命令行参数支持、多模型比较功能和完整的可视化分析工具
- 新增 `save_test_results.py` 脚本，用于将指定索引的测试样本（包括原始图像、标签和预测结果）保存为.npz文件
- 新增 `plot_results.py` 脚本，提供增强的可视化功能，可以处理保存的.npz文件并生成多种可视化图像
- 在 `plot_results.py` 中添加自动查找最新.npz文件目录的功能，无需手动指定输入路径
- 在 `plot_results.py` 中新增 `ensure_2d` 函数，用于处理不同维度的输入数组，避免广播错误
- 在 `plot_results.py` 中新增 `create_overlay_manually` 函数，手动创建覆盖图以解决类别0（上NS - 蓝色）不显示的问题
- 在 `plot_results.py` 中添加类别像素统计功能，帮助调试标签显示问题

### 修改
- 更新 `__init__.py` 文件，添加对新模型的支持
- 改进模型架构，使其能够处理任意尺寸的输入数据
- 优化 `get_model` 函数，添加对 `section_i2u_net` 的支持
- 修改 `section_train.py` 文件，将默认模型架构更改为 `section_i2u_net`
- 更新 `section_test.py` 文件，将默认模型路径更改为使用新模型
- 更新 `section_train.py` 中的梯度裁剪函数，使用非弃用的 `clip_grad_norm_` 函数
- 改进 `plot.py` 中的颜色映射和图像处理，使其更适合地震数据分析
- 将 `plot.py` 重构为 Jupyter Notebook 风格，添加多个可分块执行的示例，方便测试和可视化
- 更新 `plot_results.py` 的命令行参数处理，将 `--input` 参数设为可选，增加默认参数设置
- 调整 `plot_results.py` 的默认DPI为600，以生成更高质量的图像
- 替换 `plot_results.py` 中的 SegmentationMapsOnImage 方法，使用自定义的覆盖图生成方法，确保所有类别标签都能正确显示
- 重新创建并优化了空间金字塔池化Inception模块的结构图 (`spp_inception_block.drawio`)，使其更清晰详尽。
- 优化了spp_inception_block.drawio图表，调整了箭头线宽、跳转方式和路径，解决箭头连接关系紊乱和组件遮挡问题
- 增强了图表可读性，细化了箭头跳转样式
- 优化了`spp_inception_block.drawio`图表中的箭头连接，改进了箭头的路由方式和跳转样式，解决了箭头遮挡文字和组件标签的问题，提高了整体可读性

### 修复
- 修复了原始I2U-Net模型在处理非固定尺寸输入时的问题
- 修复了 `section_train.py` 中 `Compose()` 初始化错误，将其改为 `Compose([])`，传入空列表作为 `augmentations` 参数
- 修复了 TensorBoard 图像格式错误，为 `writer.add_image` 添加了 `dataformats='NCHW'` 参数，以匹配 `np_to_tb` 函数输出的 4 维张量格式
- 修复了验证阶段使用错误变量的问题，将 `outputs` 改为 `outputs_val`
- 修复了验证损失计算问题，使用平均验证损失而不是最后一个批次的损失
- 修复了 `section_test.py` 中的模型路径问题，使用更通用的路径格式
- 修复了 `section_test.py` 中的字典键名错误，将带冒号空格的键名（如 `'Pixel Acc: '`）更新为不带冒号空格的键名（如 `'Pixel Acc'`）
- 修复了 `section_test.py` 中的类别准确率访问错误，使用 `class_iou` 数组替代不存在的 `score["Class Accuracy"]` 列表
- 修复了 `patch_test.py` 中的TensorBoard图像显示问题，为所有 `writer.add_image` 调用添加了合适的 `dataformats` 参数
- 修复了 `plot_results.py` 中可能出现的广播错误 `ValueError: could not broadcast input array from shape (255,701) into shape (1,255)`，通过添加维度处理功能
- 修复了 `plot_results.py` 中类别0（上NS - 蓝色）在覆盖图中不显示的问题，通过实现自定义覆盖图生成算法，确保所有类别都能正确显示 