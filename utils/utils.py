
from facenet_pytorch import MTCNN
import torch
from PIL import Image


def model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


def detect_crop_image(path, model, transform, device):
    pil_image = Image.open(path)
    boxes, _ = mtcnn.detect(pil_image)
    boxes = boxes.astype(int)
    
    cropped_images = []
    for box in boxes:
        cropped_image = image.crop(box)
        cropped_images.append(transform(cropped_image))
    return torch.stack(cropped_images).to(device)