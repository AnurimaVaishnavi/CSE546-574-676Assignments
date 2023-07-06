import os

def remove_missing_images(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder_name}")
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if not os.path.isfile(image_path):
                print(f"Removing missing image: {image_name}")
                os.remove(image_path)

# Provide the path to the root folder containing subfolders with images
root_folder = "/Users/anurimavaishnavikumar/Desktop/ImageNet-Datasets-Downloader-master/data_root_folder/imagenet"
remove_missing_images(root_folder)
