# 变更日志

所有对项目的显著更改都将记录在此文件中。

## [未发布]

### 添加
- 新增 `Section_I2U_Net` 模型，支持任意尺寸输入，特别适用于Inline主测线（701×255）和Xline联络测线（401×255）整个section大小的数据
- 在 `SPP_inception_block` 中使用自适应池化层，以适应不同尺寸的输入
- 在模型前向传播中添加输入尺寸检查，确保输出尺寸与输入匹配
- 在 `section_train.py` 中添加 `model_dim` 参数，用于配置模型维度
- 在 `section_train.py` 中添加更多调试信息打印，包括数据形状、设备信息等
- 在 `section_test.py` 中添加调试信息输出，用于监控图像处理和TensorBoard记录过程
- 全面重构 `plot.py` 脚本，添加命令行参数支持、多模型比较功能和完整的可视化分析工具

### 修改
- 更新 `__init__.py` 文件，添加对新模型的支持
- 改进模型架构，使其能够处理任意尺寸的输入数据
- 优化 `get_model` 函数，添加对 `section_i2u_net` 的支持
- 修改 `section_train.py` 文件，将默认模型架构更改为 `section_i2u_net`
- 更新 `section_test.py` 文件，将默认模型路径更改为使用新模型
- 更新 `section_train.py` 中的梯度裁剪函数，使用非弃用的 `clip_grad_norm_` 函数
- 改进 `plot.py` 中的颜色映射和图像处理，使其更适合地震数据分析
- 将 `plot.py` 重构为 Jupyter Notebook 风格，添加多个可分块执行的示例，方便测试和可视化

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