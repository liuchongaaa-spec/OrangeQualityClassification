# 橘子品质分类项目

基于 PyTorch 的橘子品质二分类模型，实现了 EfficientNet-B0 + CBAM 的 OrangeNetV1 架构，以及训练/数据处理脚手架。

## 文件说明
- `model.py`：包含 EfficientNet-B0 特征提取主干、CBAM 注意力模块和 OrangeNetV1 模型（支持可选中间全连接层、可配置 Dropout 与 CBAM reduction）。
- `dataset.py`：使用 `ImageFolder` 构建训练/验证集，包含常用数据增强和 `class_indices.json` 的自动生成。
- `utils.py`：训练过程的检查点存储、加载以及精度计算和指标保存工具。
- `train.py`：训练脚本，支持超参传入，默认使用 CrossEntropyLoss + AdamW + CosineAnnealingLR。

## 快速开始
```bash
python train.py \
  --data_root /path/to/dataset \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --dropout 0.4 \
  --cbam_reduction 16 \
  --use_mid_fc
```

数据目录需包含 `train/` 与 `val/` 子目录，内部按类别分文件夹。运行后会在数据根目录下生成 `class_indices.json`，在 `outputs/` 目录下保存最佳权重与训练曲线指标。
