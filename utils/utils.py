
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import cv2
import numpy as np
import time
import os
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Subset

def model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


def detect_crop_image(model, transform, device, path=None, frame=None):
    if path:
        pil_image = Image.open(path)
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(frame)
    boxes, _ = model.detect(pil_image)
    boxes = boxes.astype(int)
    
    cropped_images = []
    for box in boxes:
        cropped_image = pil_image.crop(box)
        cropped_images.append(transform(cropped_image))
    return torch.stack(cropped_images).to(device)
    # return cropped_images

# Callback function for mouse events
def take_picture(event, x, y, flags, param):
    global take_photo, photo_time
    if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
        take_photo = True
        photo_time = time.time()

def next_folder_name(data_path=None):
    if data_path:
        folder_list = sorted(os.listdir(data_path),key=lambda x: int(x))
        if folder_list:
            return int(folder_list[-1])+1
        else:
            return 0
    else:
        print('Provide a path')

def extract_face(path=None, save_path=None):
    global take_photo
    
    next_class = next_folder_name(save_path)
    save_path = os.path.join(save_path, str(next_class))

    try:
        os.mkdir(save_path)
    except:
        pass

    if path:
        image = cv2.imread(path)
        if save_path:
            cv2.imwrite(os.path.join(save_path, '0.jpg'), image)
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
                    cv2.imwrite(os.path.join(save_path, '0.jpg'), frame)
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

    return img, save_path


def aug_img(save_path, transform, num_images):
    img = Image.open(save_path)
    partial_path = '/'.join(save_path.split('/')[:-1])
    for i in range(num_images-1):
        new_img = transform(img)
        new_img.save(os.path.join(partial_path, '{}.jpg'.format(i+1)))


def create_neg_class(path, num_images, transforms, dataset, idx_dict):
    keys = list(idx_dict.keys())
    
    for i in range(num_images):
        key = keys[i]
                
        dataset_sub = Subset(dataset, [idx_dict[key][0]])
        torch_img = dataset_sub.__getitem__(0)[0]
        pil_img = to_pil_image(torch_img)
        pil_img = transforms(pil_img)
#         print(os.path.join(path, '{}.jpg'.format(i)))
        pil_img.save(os.path.join(path, '{}.jpg'.format(i)))