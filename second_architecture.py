import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image
from PIL import Image
import numpy as np

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
        sample = (image, labels)

        return sample

# Testing Data
# dataset = TensorDataset('dataset/raw_imgs', 'dataset/masks', transform=transform)
# loader = torch.utils.data.DataLoader(dataset)
# for sample in loader:
#     print(sample['labels'].size())

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, padding=2)
        # self.conv_dilation1 = torch.nn.Conv2d(8, 8, 5, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, padding=2)
        # self.conv_dilation2 = torch.nn.Conv2d(16, 16, 5, padding=2, dilation=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)
        # self.conv_dilation3 = torch.nn.Conv2d(32, 32, 3, padding=1, dilation=1)

        self.conv4 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv_tran4 = torch.nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0)

        self.conv5 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv_tran5 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(16, 8, 5, padding=2)
        self.conv_tran6 = torch.nn.ConvTranspose2d(8, 8, 2, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(8, 2, 5, padding=2)

        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.max_pool = torch.nn.MaxPool2d(2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # This are dummy declarations
        h1 = self.max_pool(self.relu(self.conv1(x)))
        h2 = self.max_pool(self.relu(self.conv2(h1)))
        h3 = self.max_pool(self.relu(self.conv3(h2)))

        h4 = self.relu(self.conv4(h3))
        h5 = self.relu(self.conv_tran4(self.relu(self.conv4(h4))))

        h6 = self.relu(self.conv_tran5(self.relu(self.conv5(h5))))
        h7 = self.relu(self.conv_tran6(self.relu(self.conv6(h6))))
        # h7 = self.upsample(h1)
        h8 = self.conv7(h7)

        # x = F.relu(self.conv1(x))
        return h8

# Testing Softmax
# dtype = torch.cuda.FloatTensor
# inpt = Variable(torch.tensor([[[[10., 0.], [0., 0.]],[[0., 10.], [10., 10.]]]]).type(dtype))
# target = Variable(torch.tensor([[[0, 1], [1, 1]]]).type(torch.cuda.LongTensor))
# print(inpt.size())
# print(target.size())
# loss = torch.nn.CrossEntropyLoss()
# soft = loss(inpt, target)
# print(soft)


# dtype = torch.cuda.FloatTensor

model = Model().cuda()

dataset = TensorDataset('dataset/raw_imgs', 'dataset/masks', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=15)


loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 5e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(300):
    for x_batch, y_batch in loader:
        x_var = Variable(x_batch)
        y_var = Variable(y_batch)
        y_pred = model(x_var)

        loss = loss_fn(y_pred, y_var)

        optimizer.zero_grad()
        loss.backward()

        # print(loss.data[0])
        
        optimizer.step()
    print("Epoch:",epoch,"Loss:",loss.data[0])
import pickle
with open('second_model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

# def show_img_2(pt_tensor):
#     img_r = pt_tensor[0][1]
#     img_b = pt_tensor[0][0]
#     img_g = img_r - img_b
#     img_g[img_r > img_b] = 1

    
#     print(torch.sum(img_g))


validation_dataset = TensorDataset('dataset/validation_imgs', 'dataset/validation_masks', transform=transform)
loader = torch.utils.data.DataLoader(validation_dataset)

for x_batch, y_batch in loader:
    x_var = Variable(x_batch)
    y_var = Variable(y_batch)
    y_val_pred = model(x_var)
    loss_val = loss_fn(y_val_pred, y_var)

print("Val loss:", loss_val)
show_img(y_val_pred)