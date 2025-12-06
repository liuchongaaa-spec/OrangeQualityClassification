import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import OrangeNetV1


# ================================
# Grad-CAM 目标层（EfficientNet-B0 最末层）
# ================================
def get_target_layer(model):
    return model.backbone.features[-1]


# ================================
# 单图预测
# ================================
def predict_one_image(model, img_path, class_dict, device):
    model.eval()

    # --- 1. 读图 ---
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0

    transform_no_norm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    input_tensor = transform_no_norm(img).unsqueeze(0).to(device)

    # --- 2. 预测 ---
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    # --- 3. 生成 Grad-CAM ---
    target_layer = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device != "cpu"))

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # --- 4. 显示 ---
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Predict: {class_dict[str(pred_class)]} (prob={probs[pred_class]:.3f})")

    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()

    return pred_class, probs[pred_class]


# ================================
# 主函数
# ================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device.")

    # ---- 配置路径 ----
    img_path = "./pic/img.png"
    weights_path = "./outputs/best.pth"     # ← 训练出的 best.pth
    class_json = "./class_indices.json"

    assert os.path.exists(img_path), f"{img_path} not found."
    assert os.path.exists(weights_path), f"{weights_path} not found."
    assert os.path.exists(class_json), "class_indices.json not found."

    # ---- 加载 class_indices.json ----
    with open(class_json, "r") as f:
        class_dict = json.load(f)
    num_classes = len(class_dict)

    # ---- 构建模型 ----
    model = OrangeNetV1(
        num_classes=num_classes,
        dropout_rate=0.4,
        cbam_reduction_ratio=16,
        use_mid_fc=False
    ).to(device)

    # ---- 加载训练权重（不包含 'model' 键）----
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded model weights.")

    # ---- 预测 ----
    pred_class, prob = predict_one_image(model, img_path, class_dict, device)

    print("\n===== Final Result =====")
    print(f"Class: {class_dict[str(pred_class)]}")
    print(f"Prob : {prob:.4f}")


if __name__ == "__main__":
    main()
