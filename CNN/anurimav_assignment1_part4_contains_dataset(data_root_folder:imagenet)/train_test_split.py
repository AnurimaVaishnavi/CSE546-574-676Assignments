import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models


def preprocess_dataset(root_folder, test_size=0.2, val_size=0.2):
    image_paths = []
    labels = []
    
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder_name}")
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(folder_name) 

    normalized_images = []
    scaler = MinMaxScaler()

    for image_path in image_paths:
        img = Image.open(image_path)
        img = img.resize((256, 256))
        img_array = np.array(img)
        normalized_img = scaler.fit_transform(img_array)
        normalized_images.append(normalized_img)

    X_train, X_test, y_train, y_test = train_test_split(
        normalized_images, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test



# root_folder = "/Users/anurimavaishnavikumar/Desktop/ImageNet-Datasets-Downloader-master/data_root_folder/imagenet"
# X_train, X_val, X_test, y_train, y_val, y_test = preprocess_dataset(root_folder)

