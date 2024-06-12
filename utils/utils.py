
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import cv2
import numpy as np
import time


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

# Callback function for mouse events
def take_picture(event, x, y, flags, param):
    global take_photo, photo_time
    if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
        take_photo = True
        photo_time = time.time()

def extract_face(path=None, save_path=None):
    global take_photo
    
    if path:
        image = cv2.imread(path)
        if save_path:
            cv2.imwrite(save_path, image)
        else:
            cv2.imwrite('photo.jpg', image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Initialize the camera
        cap = cv2.VideoCapture(0)

        # Set the mouse callback function for the window
        cv2.namedWindow('Camera')
        cv2.setMouseCallback('Camera', take_picture)
        take_photo = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Camera', frame)

            if take_photo:
                if save_path:
                    cv2.imwrite(save_path, frame)
                else:
                    cv2.imwrite('photo.jpg', frame)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                print("Photo taken!")
                take_photo = False
                
                time.sleep(1)
                break

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

    return img