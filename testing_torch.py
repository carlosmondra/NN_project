import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,transform=transforms.ToTensor()):
        """
        Args:
        root_dir(string): directory with all train images
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        str_idx = str(idx + 1)
        img_name = ('0' * (7 - len(str_idx) + 1)) + str_idx + '.png'
        img_path = os.path.join(self.root_dir, img_name)
        image = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        #resides in 'NN_project/dataset/masks'
        mask_path = os.path.join(self.root_dir, img_name)
        mask = PIL.Image.open(img_path).convert('RGB')
        mask = np.array(mask)
        mask = mask.transpose((2,0,1)) #transpose ot get color channel as first dimension
        mask_road = mask[0]
        
        label_road = (mask_road[:][:]==255)*1
        label_empty = (mask_road[:][:]!=255)*1
        
        labels = np.array([label_road, label_empty]).type(torch.cuda.LongTensor)
        
        #labels = torch.randint(2, (2,640,640)).type(torch.cuda.LongTensor)
        sample = {'image': image, 'labels': labels}

        return sample

# Testing Data
# dataset = TensorDataset('dataset', transform=transform)
# loader = torch.utils.data.DataLoader(dataset, batch_size=20)
# for sample in loader:
#     print(sample['image'].size())

import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)

        self.conv4 = torch.nn.Conv2d(32, 32, 3, padding=1)

        self.conv5 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 8, 5, padding=2)
        self.conv7 = torch.nn.Conv2d(8, 2, 5, padding=2)

        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.max_pool = torch.nn.MaxPool2d(2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # This are dummy declarations
        h1 = self.max_pool(self.relu(self.conv1(x)))
        # h2 = self.max_pool(self.relu(self.conv2(h1)))
        # h3 = self.max_pool(self.relu(self.conv2(h2)))
        
        # h4 = self.relu(self.conv4(h3))
        # h5 = self.upsample(self.relu(self.conv4(h4)))

        # h6 = self.upsample(self.relu(self.conv5(h5)))
        # h7 = self.upsample(self.relu(self.conv6(h6)))
        h7 = self.upsample(h1)
        h8 = self.relu(self.conv6(h7))

        # x = F.relu(self.conv1(x))
       return h8

# Testing Softmax
# dtype = torch.cuda.FloatTensor
# inpt = Variable(torch.tensor([[[[10., 0.], [0., 0.]],[[0., 10.], [10., 10.]]]]).type(dtype))
# target = Variable(torch.tensor([[[0, 1], [1, 1]]]).type(torch.cuda.LongTensor))
# loss = torch.nn.CrossEntropyLoss()
# soft = loss(inpt, target)
# print(soft)


# dtype = torch.cuda.FloatTensor

# dataset = TensorDataset('dataset', transform=transform)
# loader = torch.utils.data.DataLoader(dataset, batch_size=20)
# model = Model()

# # Check CrossEntropy options
# loss = torch.nn.CrossEntropyLoss()
# learning_rate = 1e-4
# # I haven't check the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(10)
#     for x_batch, y_batch in loader:
#         x_var, y_var = Variable(x_batch), Variable(y_batch)
#         y_pred = model(x_var)
#         loss = loss_fn(y_pred, y_var)

#         optimizer.zero_grad()
#         loss.backward()
        
#         optimizer.step()