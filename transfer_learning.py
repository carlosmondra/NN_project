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


n_class    = 2

batch_size = 3
epochs     = 90
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)


# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
# num_gpu = list(range(torch.cuda.device_count()))

# if sys.argv[1] == 'CamVid':
#     train_data = CamVidDataset(csv_file=train_file, phase='train')
# else:
#     train_data = CityscapesDataset(csv_file=train_file, phase='train')
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

# if sys.argv[1] == 'CamVid':
#     val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
# else:
#     val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)
# val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

vgg_model = VGGNet(requires_grad=True, remove_fc=True, model='vgg16')
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    # fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self,raw_imgs_dir,masks_dir,transform=transforms.ToTensor()):
        """
        Args:
        raw_imgs_dir(string): directory with all train images
        """
        self.raw_imgs_dir = raw_imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len([name for name in os.listdir(self.raw_imgs_dir) if os.path.isfile(os.path.join(self.raw_imgs_dir, name))])

    def __getitem__(self, idx):
        str_idx = str(idx + 1)
        img_name = ('0' * (7 - len(str_idx) + 1)) + str_idx + '.png'
        img_path = os.path.join(self.raw_imgs_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(image).type(torch.cuda.FloatTensor)
        
        #resides in 'NN_project/dataset/masks'
        mask_path = os.path.join(self.masks_dir, img_name)
        mask = PIL.Image.open(mask_path).convert('RGB')
        mask = transforms.ToTensor()(mask).type(torch.cuda.LongTensor)
        labels = mask[0]
        # sample = (image, labels)
        sample = {'X':image, 'Y':labels}
        return sample

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
# score_dir = os.path.join("scores", configs)
# if not os.path.exists(score_dir):
#     os.makedirs(score_dir)
# IU_scores    = np.zeros((epochs, n_class))
# pixel_scores = np.zeros(epochs)

dataset = TensorDataset('dataset/raw_imgs', 'dataset/masks', transform=transform)
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

        # val(epoch)


# def val(epoch):
#     fcn_model.eval()
#     total_ious = []
#     pixel_accs = []
#     for iter, batch in enumerate(val_loader):
#         if use_gpu:
#             inputs = Variable(batch['X'].cuda())
#         else:
#             inputs = Variable(batch['X'])

#         output = fcn_model(inputs)
#         output = output.data.cpu().numpy()

#         N, _, h, w = output.shape
#         pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

#         target = batch['l'].cpu().numpy().reshape(N, h, w)
#         for p, t in zip(pred, target):
#             total_ious.append(iou(p, t))
#             pixel_accs.append(pixel_acc(p, t))

#     # Calculate average IoU
#     total_ious = np.array(total_ious).T  # n_class * val_len
#     ious = np.nanmean(total_ious, axis=1)
#     pixel_accs = np.array(pixel_accs).mean()
#     print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
#     IU_scores[epoch] = ious
#     np.save(os.path.join(score_dir, "meanIU"), IU_scores)
#     pixel_scores[epoch] = pixel_accs
#     np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# # borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# # Calculates class intersections over unions
# def iou(pred, target):
#     ious = []
#     for cls in range(n_class):
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = pred_inds[target_inds].sum()
#         union = pred_inds.sum() + target_inds.sum() - intersection
#         if union == 0:
#             ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
#         else:
#             ious.append(float(intersection) / max(union, 1))
#         # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
#     return ious


# def pixel_acc(pred, target):
#     correct = (pred == target).sum()
#     total   = (target == target).sum()
#     return correct / total

def show_img(pt_tensor):
    y_pred_np = pt_tensor.cpu().detach().numpy()
    #get the first image and the red color channel
    img_r = y_pred_np[0][1]
    img_b = y_pred_np[0][0]
    img_g = (img_r > img_b) * 255
    img = np.array([img_g, np.zeros((640,640)), np.zeros((640,640))]).astype(np.uint8)

    img = img.transpose(1,2,0)
    img = Image.fromarray(img, 'RGB')
    img.save('preview.png')
    img.show()

def show_masked_img(pt_tensor, raw_img):
    y_pred_np = pt_tensor.cpu().detach().numpy()
    img_r = y_pred_np[0][1]
    img_b = y_pred_np[0][0]
    pred = img_r > img_b
    img_array = np.array(raw_img).astype(np.uint8)
    red_array = np.zeros_like(img_array[:,:,0]) + 255
    img_array[:,:,0] //= 2

    mask = red_array - img_array[:,:,0]

    mask[mask > 255] = 255
    mask *= pred.astype(np.uint8)
    img_array[:,:,0] += mask
    img_array[:,:,1] //= 2
    img_array[:,:,2] //= 2
    image = Image.fromarray(img_array, 'RGB')
    image.save('preview2.png')
    image.show()

if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    # train()

    fcn_model = torch.load("models/FCNs-BCEWithLogits_batch3_epoch90_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05")

    validation_dataset = TensorDataset('dataset/validation_imgs', 'dataset/validation_masks', transform=transform)
    loader = torch.utils.data.DataLoader(validation_dataset)

    for batch in loader:
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())
        else:
            inputs, labels = Variable(batch['X']), Variable(batch['Y'])
        y_val_pred = fcn_model(inputs)
        loss_val = criterion(y_val_pred, labels)

    print("Val loss:", loss_val.item())
    # show_img(y_val_pred)
    raw_img = PIL.Image.open("dataset/validation_imgs/00000001.png").convert('RGB')
    show_masked_img(y_val_pred, raw_img)
