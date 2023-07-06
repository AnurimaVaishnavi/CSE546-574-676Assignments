from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models


def normalize_images(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder_name}")
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                # Normalize the image (e.g., resize, convert to grayscale, etc.)
                img = Image.open(image_path)
                
                # Replace the following lines with your desired normalization techniques
                
                img = img.resize((224, 224))  # Resize the image to a specific size
                img = img.convert('L')  # Convert the image to grayscale
                
                # Save the normalized image, overwrite the original image if desired
                img.save(image_path)


root_folder = "/Users/anurimavaishnavikumar/Desktop/ImageNet-Datasets-Downloader-master/data_root_folder/imagenet"
normalize_images(root_folder)
