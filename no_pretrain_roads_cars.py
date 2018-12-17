import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image
from PIL import Image
import numpy as np

import argparse
parser = argparse.ArgumentParser(description="This program shows how the first and second architectures work.")
parser.add_argument('architecture', choices=['1','2'], help="Choose the architecture to use.")
parser.add_argument('-v', '--validate', action='store_true', help="When passing v, a model will be loaded and validated using validation images. If this argument is not passed, then a new model will be trained and stored in the models folder.")
parser.add_argument('-p', '--persist', action='store_true', help="Persist image.")
parser.add_argument('-s', '--show', action='store_true', help="Show image.")

args = parser.parse_args()

raw_imgs_dir = 'dataset/raw_imgs'
masks_dir = 'dataset/masks'
validation_imgs = 'dataset/validation_imgs'
validation_masks = 'dataset/validation_masks'
pred_imgs = 'dataset/road_pred_imgs'

if args.architecture == '1':
    model_name = "First_Architecture"
    predictions_path = 'dataset/first_road_preds/prediction_'
else:
    model_name = "Second_Architecture"
    predictions_path = 'dataset/second_road_preds/prediction_'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Testing Softmax
# dtype = torch.cuda.FloatTensor
# inpt = Variable(torch.tensor([[[[10., 0.], [0., 0.]],[[0., 10.], [10., 10.]]]]).type(dtype))
# target = Variable(torch.tensor([[[0, 1], [1, 1]]]).type(torch.cuda.LongTensor))
# print(inpt.size())
# print(target.size())
# loss = torch.nn.CrossEntropyLoss()
# soft = loss(inpt, target)
# print(soft)

if args.architecture == '1':
    from lib import FirstModel
    model = FirstModel().cuda()
else:
    from lib import SecondModel
    model = SecondModel().cuda()

from lib import TensorDataset
dataset = TensorDataset(raw_imgs_dir, masks_dir, "roads", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=15)


loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 5e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if args.validate:
    model = torch.load('models/' + model_name)
else:
    for epoch in range(200):
        for batch in loader:

            x_var = Variable(batch['X'].cuda())
            y_var = Variable(batch['Y'].cuda())
            y_pred = model(x_var)

            loss = loss_fn(y_pred, y_var)

            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
        print("Epoch:",epoch,"Loss:",loss.item())

    torch.save(model, 'models/' + model_name)

def show_img(pt_tensor):
    y_pred_np = pt_tensor.cpu().detach().numpy()
    #get the first image and the red color channel
    img_r = y_pred_np[0][1]
    img_b = y_pred_np[0][0]
    img_g = (img_r > img_b) * 255
    img = np.array([img_g, np.zeros((640,640)), np.zeros((640,640))]).astype(np.uint8)

    img = img.transpose(1,2,0)
    img = Image.fromarray(img, 'RGB')
    # img.save('preview.png')
    img.show()

validation_dataset = TensorDataset(validation_imgs, validation_masks, "roads", transform=transform)
loader = torch.utils.data.DataLoader(validation_dataset)

for batch in loader:
    x_var = Variable(batch['X'].cuda())
    y_var = Variable(batch['Y'].cuda())
    y_val_pred = model(x_var)
    loss_val = loss_fn(y_val_pred, y_var)

print("Val loss:", loss_val)
show_img(y_val_pred)