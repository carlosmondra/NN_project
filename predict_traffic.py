import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import PIL.Image
from PIL import Image

from fully_conv_nets import VGGNet, FCNs#, FCN32s, FCN16s, FCN8s

import numpy as np
import time
import sys
import os

use_gpu = torch.cuda.is_available()

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self,raw_imgs_dir,masks_dir,data_type,transform=transforms.ToTensor()):
        """
        Args:
        raw_imgs_dir(string): directory with all train images
        """
        self.raw_imgs_dir = raw_imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.data_type = 0
        if data_type == "cars":
            self.data_type = 2

    def __len__(self):
        return len([name for name in os.listdir(self.raw_imgs_dir) if os.path.isfile(os.path.join(self.raw_imgs_dir, name))])

    def __getitem__(self, idx):
        str_idx = str(idx + 1)
        img_name = ('0' * (7 - len(str_idx) + 1)) + str_idx + '.png'
        img_path = os.path.join(self.raw_imgs_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(image).type(torch.cuda.FloatTensor)
        
        #resides in 'NN_project/dataset/car_masks'
        mask_path = os.path.join(self.masks_dir, img_name)
        mask = PIL.Image.open(mask_path).convert('RGB')
        mask = transforms.ToTensor()(mask).type(torch.cuda.LongTensor)
        labels = mask[self.data_type]
        # sample = (image, labels)
        sample = {'X':image, 'Y':labels}
        return sample

def show_masked_img(roads_tensor, cars_tensor, raw_img):
    # Convert to numpy
    roads_pred = roads_tensor.cpu().detach().numpy()
    cars_pred = cars_tensor.cpu().detach().numpy()
    # Extract roads
    roads_img_r = roads_pred[1]
    roads_img_b = roads_pred[0]
    # Extract cars
    cars_img_r = cars_pred[1]
    cars_img_b = cars_pred[0]

    # Setting true valeus where the pixel is more likely to be a road
    roads_pred = roads_img_r > roads_img_b
    # Setting true valeus where the pixel is more likely to be a car
    cars_pred = cars_img_r > cars_img_b
    # Only care about the cars predicted on roads
    cars_pred *= roads_pred
    # Remove car pixels from road pixels
    roads_pred *= np.logical_not(cars_pred)

    # Convert raw_img to numpy array
    img_array = np.array(raw_img).astype(np.uint8)
    # Make the image more opaque
    # img_array //= 3
    # img_array *= 2
    # Create array of 255
    colored_array = np.zeros_like(img_array[:,:,0]) + 255

    # Creating masks
    roads_mask = colored_array - img_array[:,:,0]
    cars_mask = colored_array - img_array[:,:,2]
    roads_mask *= roads_pred.astype(np.uint8)
    cars_mask *= cars_pred.astype(np.uint8)

    # Create image
    img_array[:,:,0] += roads_mask
    img_array[:,:,2] += cars_mask
    image = Image.fromarray(img_array, 'RGB')
    # image.save('preview2.png')
    image.show()

def get_traffic_level(roads_tensor, cars_tensor):
    # Convert to numpy
    roads_pred = roads_tensor.cpu().detach().numpy()
    cars_pred = cars_tensor.cpu().detach().numpy()
    # Extract roads
    roads_img_r = roads_pred[1]
    roads_img_b = roads_pred[0]
    # Extract cars
    cars_img_r = cars_pred[1]
    cars_img_b = cars_pred[0]

    # Setting true valeus where the pixel is more likely to be a road
    roads_pred = roads_img_r > roads_img_b
    # Setting true valeus where the pixel is more likely to be a car
    cars_pred = cars_img_r > cars_img_b
    # Only care about the cars predicted on roads
    cars_pred *= roads_pred

    # Check if enough roads
    road_pixels = np.sum(roads_pred)
    total_pixels = roads_pred.shape[0] * roads_pred.shape[1]

    if road_pixels / total_pixels < 0.1:
        return "Not enough roads"
    
    #Check number of car pixels
    car_pixels = np.sum(cars_pred)
    traffic = car_pixels / road_pixels
    # print("Traffic", traffic)
    if traffic < 0.2:
        return "No traffic"
    if traffic < 0.5:
        return "Moderate traffic"
    if traffic < 0.8:
        return "Traffic"
    return "Heavy traffic"

if __name__ == "__main__":
    import pickle
    directory = 'dataset/car_preds'
    num_of_files = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    
    for idx in range(1, num_of_files + 1):
        str_idx = str(idx)

        with open('dataset/car_preds/prediction_' + str_idx + '.pred', 'rb') as handle:
            cars_pred, _ = pickle.load(handle)

        with open('dataset/road_preds/prediction_' + str_idx + '.pred', 'rb') as handle:
            roads_pred, raw_img = pickle.load(handle)

        show_masked_img(roads_pred, cars_pred, raw_img)
        # print(get_traffic_level(roads_pred, cars_pred))