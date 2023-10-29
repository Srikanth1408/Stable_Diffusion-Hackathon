import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import os
import gc


dest = "/home/harshabommana/idd20kII/sam_generated_masks"
source =  "/home/harshabommana/idd20kII/leftImg8bit/train"

def show_masks_on_image( masks,save_path,random_color=True):


    for mask in masks:
        if random_color:
            color = np.concatenate(
                [np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)
    
    plt.axis('off')
    # print(save_path)
    plt.savefig(save_path)
    del masks
    gc.collect()


def predict(folder,file):    

    file_name = os.path.join(source,folder,file)
    img = Image.open(file_name).convert("RGB")
    print(file_name)
    generator = pipeline("mask-generation", model="facebook/sam-vit-large", device=0)
    outputs = generator(img, points_per_batch=8)
    masks = outputs["masks"]
    dest_path = os.path.join(dest,folder,file)
    print(dest_path)
    show_masks_on_image( masks,dest_path)


for folder in os.listdir(source):
    if folder in ["201","333","410","473","475"]:
        continue
    folder_name = os.path.join(source,folder)
    print(folder_name)
    if os.path.isdir(folder_name):
        destination_folder = os.path.join(dest,folder)
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)

        for file in os.listdir(folder_name):
            file_name = os.path.join(folder_name,file)
            predict(folder,file)
        
            
    