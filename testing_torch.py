import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image

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

        labels = torch.randint(2, (2,640,640)).type(torch.cuda.LongTensor)
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
        # This are dummy declarations
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        # This are dummy declarations
        x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

# Testing Softmax
# dtype = torch.cuda.FloatTensor
# inpt = Variable(torch.tensor([[[[10., 0.], [0., 0.]],[[0., 10.], [10., 10.]]]]).type(dtype))
# target = Variable(torch.tensor([[[0, 1], [1, 1]]]).type(torch.cuda.LongTensor))
# loss = torch.nn.CrossEntropyLoss()
# soft = loss(inpt, target)
# print(soft)


# dtype = torch.cuda.FloatTensor

dataset = TensorDataset('dataset', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=20)
model = Model()

# Check CrossEntropy options
loss = torch.nn.CrossEntropyLoss()
learning_rate = 1e-4
# I haven't check the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(10)
    for x_batch, y_batch in loader:
        x_var, y_var = Variable(x_batch), Variable(y_batch)
        y_pred = model(x_var)
        loss = loss_fn(y_pred, y_var)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()