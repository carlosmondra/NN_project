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

    # Convert raw_img to numpy array
    img_array = np.array(raw_img).astype(np.uint8)
    # Make the image more opaque
    img_array //= 2
    # Create array of 255
    colored_array = np.zeros_like(img_array[:,:,0]) + 255
    # img_array[:,:,0] //= 2

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

if __name__ == "__main__":
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    roads_model = torch.load("models/FCNs-BCEWithLogits_batch3_epoch90_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05")
    roads_val_dataset = TensorDataset('dataset/validation_imgs', 'dataset/validation_masks', "roads", transform=validation_transform)

    roads_loader = torch.utils.data.DataLoader(roads_val_dataset)

    road_predictions = []

    for idx, roads_batch in enumerate(roads_loader):
        if use_gpu:
            road_inputs = Variable(roads_batch['X'].cuda())
            road_labels = Variable(roads_batch['Y'].cuda())
        else:
            road_inputs, road_labels = Variable(roads_batch['X']), Variable(roads_batch['Y'])
        roads_pred = roads_model(road_inputs)
        road_predictions.append(roads_pred)

    torch.cuda.empty_cache()

    cars_model = torch.load("models/cars-CrossEnt_batch2_epoch100_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05")
    cars_val_dataset = TensorDataset('dataset/validation_imgs', 'dataset/validation_masks', "cars", transform=validation_transform)

    cars_loader = torch.utils.data.DataLoader(cars_val_dataset)

    car_predictions = []

    for idx, cars_batch in enumerate(cars_loader):
        if use_gpu:
            car_inputs = Variable(cars_batch['X'].cuda())
            car_labels = Variable(cars_batch['Y'].cuda())
        else:
            car_inputs, car_labels = Variable(cars_batch['X']), Variable(cars_batch['Y'])
        cars_pred = cars_model(road_inputs)
        car_predictions.append(cars_pred)

    torch.cuda.empty_cache()

    for idx, (roads_pred, cars_pred) in enumerate(zip(road_predictions, car_predictions)):
        str_idx = str(idx + 1)
        img_name = ('0' * (7 - len(str_idx) + 1)) + str_idx + '.png'
        raw_img = PIL.Image.open("dataset/validation_imgs/" + img_name).convert('RGB')
        show_masked_img(roads_pred[0], cars_pred[0], raw_img)