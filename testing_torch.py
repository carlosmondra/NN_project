import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
import PIL.Image
# import pandas as pd

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

class TensorDataset(torch.utils.data.Dataset):

    # def __init__(self,text_file,root_dir,transform=transforms.ToTensor()):
    def __init__(self,root_dir,transform=transforms.ToTensor()):
        """
        Args:
        text_file(string): path to text file
        root_dir(string): directory with all train images
        """
        # self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
        # self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        if idx == 0:
            img_name = os.path.join(self.root_dir, '00000001.png')
        else:
            img_name = os.path.join(self.root_dir, '00000002.png')
        # img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = PIL.Image.open(img_name).convert('RGB')
        image = self.transform(image)
        # labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        labels = torch.randint(2, (2,640,640)).type(torch.cuda.LongTensor)
        sample = {'image': image, 'labels': labels}

        return sample

dataset = TensorDataset('dataset')
loader = torch.utils.data.DataLoader(dataset)

for sample in loader:
    print(sample['image'].size())

# def load_dataset():
#     data_path = 'data/train/'
#     train_dataset = torchvision.datasets.ImageFolder(
#         root=data_path,
#         transform=torchvision.transforms.ToTensor()
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=64,
#         num_workers=0,
#         shuffle=True
#     )
#     return train_loader

dtype = torch.cuda.FloatTensor

inpt = Variable(torch.tensor([[[[10., 0.], [0., 0.]],[[0., 10.], [10., 10.]]]]).type(dtype))
target = Variable(torch.tensor([[[0, 1], [1, 1]]]).type(torch.cuda.LongTensor))
# target = Variable(torch.randint(2, (1,2,2)).type(torch.cuda.LongTensor))
print(target.size())
# print(x.size())

# x = Variable(torch.randn(1, 2, ).type(dtype))
# y = Variable(torch.randn(1, 2, ).type(dtype), requires_grad=False)

loss = torch.nn.CrossEntropyLoss()
soft = loss(inpt, target)

print(soft)

# dtype = torch.cuda.FloatTensor

# N, D_in, H, D_out = 64, 1000, 100, 10

# x = Variable(torch.randint(2, (1,1,2,2)).type(torch.cuda.LongTensor))
# y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)
# print(x)

# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out)
# )
# loss_fn = torch.nn.MSELoss(reduction='sum')
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for t in range(500):
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y)

#     optimizer.zero_grad()
#     loss.backward()
    
#     optimizer.step()