import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
)


def stylize_frame(frame, model, device):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).cpu()

    output = output.squeeze(0).clamp(0, 255).numpy()
    output = output.transpose(1, 2, 0).astype("uint8")
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output_bgr


def stylize_video(input_path, output_path, model, device, target_width=640):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tính tỉ lệ resize theo target_width giữ tỉ lệ
    scale = target_width / original_w
    target_height = int(original_h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame trước khi stylize để giảm tải
        frame_resized = cv2.resize(frame, (target_width, target_height))

        styled_frame = stylize_frame(frame_resized, model, device)

        out.write(styled_frame)

    cap.release()
    out.release()
