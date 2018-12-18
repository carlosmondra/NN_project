import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
from fully_conv_nets import VGGNet, FCNs

import argparse
parser = argparse.ArgumentParser(description="This program trains a Neural Network to detect either cars or roads. It can also load a pretrained model to predict new roads and cars.")
parser.add_argument('type', choices=['roads','cars'], help="Choose the type of model to train/load.")
parser.add_argument('-v', '--validate', action='store_true', help="When passing v, a model will be loaded and validated using validation images. If this argument is not passed, then a new model will be trained and stored in the models folder.")
parser.add_argument('-p', '--persist', action='store_true', help="Persist image.")
parser.add_argument('-s', '--show', action='store_true', help="Show image.")

args = parser.parse_args()

n_class = 2
batch_size = 2
epochs = 200
lr = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 50
gamma = 0.5

if args.type == 'roads':
    configs = "roads-CrossEnt_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
    raw_imgs_dir = 'dataset/raw_imgs'
    masks_dir = 'dataset/masks'
    model_to_load = "FCNs-BCEWithLogits_batch3_epoch90_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"
    validation_imgs = 'dataset/validation_imgs'
    validation_masks = 'dataset/validation_masks'
    predictions_path = 'dataset/road_preds/prediction_'
    pred_imgs = 'dataset/road_pred_imgs'
else:
    configs = "cars-CrossEnt_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
    raw_imgs_dir = 'dataset/car_raw_imgs'
    masks_dir = 'dataset/car_masks'
    model_to_load = "cars-CrossEnt_batch2_epoch100_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05"
    validation_imgs = 'dataset/validation_imgs'
    validation_masks = 'dataset/validation_masks'
    predictions_path = 'dataset/car_preds/prediction_'
    pred_imgs = 'dataset/car_pred_imgs'

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()

vgg_model = VGGNet(requires_grad=True, remove_fc=True, model='vgg16')
from lib import LastModel
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class, last_layer=LastModel(32, n_class))

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

training_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
])
from lib import TensorDataset
dataset = TensorDataset(raw_imgs_dir, masks_dir, args.type, transform=training_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

if __name__ == "__main__":
    if args.validate:
        fcn_model = torch.load("models/" + model_to_load)
    else:
        train()

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    validation_dataset = TensorDataset(validation_imgs, validation_masks, args.type, transform=validation_transform)
    loader = torch.utils.data.DataLoader(validation_dataset)

    import pickle
    from lib import get_simple_masked_img

    for idx, batch in enumerate(loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())
        else:
            inputs, labels = Variable(batch['X']), Variable(batch['Y'])
        y_val_pred = fcn_model(inputs)

        str_idx = str(idx + 1)
        img_name = ('0' * (7 - len(str_idx) + 1)) + str_idx + '.png'
        raw_img = Image.open(validation_imgs + "/" + img_name).convert('RGB')
        get_simple_masked_img(y_val_pred[0], raw_img, pred_imgs, img_name, args.persist, args.show)
        with open(predictions_path + str_idx + '.pred', 'wb') as handle:
            pickle.dump((y_val_pred[0], raw_img), handle, protocol=pickle.HIGHEST_PROTOCOL)