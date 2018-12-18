import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

class FirstModel(torch.nn.Module):
    def __init__(self):
        super(FirstModel, self).__init__()
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
        h2 = self.max_pool(self.relu(self.conv2(h1)))
        h3 = self.max_pool(self.relu(self.conv3(h2)))

        h4 = self.relu(self.conv4(h3))
        h5 = self.upsample(self.relu(self.conv4(h4)))

        h6 = self.upsample(self.relu(self.conv5(h5)))
        h7 = self.upsample(self.relu(self.conv6(h6)))
        # h7 = self.upsample(h1)
        h8 = self.conv7(h7)

        # x = F.relu(self.conv1(x))
        return h8

class SecondModel(torch.nn.Module):
    def __init__(self):
        super(SecondModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, padding=2)
        # self.conv_dilation1 = torch.nn.Conv2d(8, 8, 5, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, padding=2)
        # self.conv_dilation2 = torch.nn.Conv2d(16, 16, 5, padding=2, dilation=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)
        # self.conv_dilation3 = torch.nn.Conv2d(32, 32, 3, padding=1, dilation=1)

        self.conv4 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv_tran4 = torch.nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0)

        self.conv5 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv_tran5 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(16, 8, 5, padding=2)
        self.conv_tran6 = torch.nn.ConvTranspose2d(8, 8, 2, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(8, 2, 5, padding=2)
        
        self.max_pool = torch.nn.MaxPool2d(2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # This are dummy declarations
        # pre_h1 = self.relu(self.conv_dilation1(self.relu(self.conv1(x))))
        pre_h1 = self.relu(self.conv1(x))
        h1 = self.max_pool(pre_h1)
        # pre_h2 = self.relu(self.conv_dilation2(self.relu(self.conv2(h1))))
        pre_h2 = self.relu(self.conv2(h1))
        h2 = self.max_pool(pre_h2)
        # pre_h3 = self.relu(self.conv_dilation3(self.relu(self.conv3(h2))))
        pre_h3 = self.relu(self.conv3(h2))
        h3 = self.max_pool(pre_h3)

        h4 = self.relu(self.conv4(h3))
        h5 = self.relu(self.conv_tran4(self.relu(self.conv4_2(h4))))

        h6 = self.relu(self.conv_tran5(self.relu(self.conv5(h5))))
        h7 = self.relu(self.conv_tran6(self.relu(self.conv6(h6))))
        h8 = self.conv7(h7)

        # x = F.relu(self.conv1(x))
        return h8

class LastModel(torch.nn.Module):
    def __init__(self, input_depth, output_depth):
        super(LastModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_depth, 8, 5, padding=2)
        # self.conv_dilation1 = torch.nn.Conv2d(8, 8, 5, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, padding=2)
        # self.conv_dilation2 = torch.nn.Conv2d(16, 16, 5, padding=2, dilation=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, padding=1)
        # self.conv_dilation3 = torch.nn.Conv2d(32, 32, 3, padding=1, dilation=1)

        self.conv4 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv_tran4 = torch.nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0)

        self.conv5 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv_tran5 = torch.nn.ConvTranspose2d(16, 16, 2, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(16, 8, 5, padding=2)
        self.conv_tran6 = torch.nn.ConvTranspose2d(8, 8, 2, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(8, output_depth, 5, padding=2)
        
        self.max_pool = torch.nn.MaxPool2d(2, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # This are dummy declarations
        # pre_h1 = self.relu(self.conv_dilation1(self.relu(self.conv1(x))))
        pre_h1 = self.relu(self.conv1(x))
        h1 = self.max_pool(pre_h1)
        # pre_h2 = self.relu(self.conv_dilation2(self.relu(self.conv2(h1))))
        pre_h2 = self.relu(self.conv2(h1))
        h2 = self.max_pool(pre_h2)
        # pre_h3 = self.relu(self.conv_dilation3(self.relu(self.conv3(h2))))
        pre_h3 = self.relu(self.conv3(h2))
        h3 = self.max_pool(pre_h3)

        h4 = self.relu(self.conv4(h3))
        h5 = self.relu(self.conv_tran4(self.relu(self.conv4_2(h4))))

        h6 = self.relu(self.conv_tran5(self.relu(self.conv5(h5))))
        h7 = self.relu(self.conv_tran6(self.relu(self.conv6(h6))))
        h8 = self.conv7(h7)

        # x = F.relu(self.conv1(x))
        return h8

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
        if data_type == 'cars':
            self.data_type = 2

    def __len__(self):
        return len([name for name in os.listdir(self.raw_imgs_dir) if os.path.isfile(os.path.join(self.raw_imgs_dir, name))])

    def __getitem__(self, idx):
        str_idx = str(idx + 1)
        img_name = ('0' * (7 - len(str_idx) + 1)) + str_idx + '.png'
        img_path = os.path.join(self.raw_imgs_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image).type(torch.cuda.FloatTensor)
        
        #resides in 'NN_project/dataset/car_masks'
        mask_path = os.path.join(self.masks_dir, img_name)
        mask = Image.open(mask_path).convert('RGB')
        mask = transforms.ToTensor()(mask).type(torch.cuda.LongTensor)
        labels = mask[self.data_type]
        # sample = (image, labels)
        sample = {'X':image, 'Y':labels}
        return sample

def get_simple_masked_img(pt_tensor, raw_img, pred_imgs, img_name, persist=False, show=False):
    y_pred_np = pt_tensor.cpu().detach().numpy()
    img_r = y_pred_np[1]
    img_b = y_pred_np[0]
    pred = img_r > img_b
    img_array = np.array(raw_img).astype(np.uint8)
    red_array = np.zeros_like(img_array[:,:,0]) + 255
    img_array //= 2

    mask = red_array - img_array[:,:,0]

    mask[mask > 255] = 255
    mask *= pred.astype(np.uint8)
    img_array[:,:,0] += mask
    image = Image.fromarray(img_array, 'RGB')
    if persist:
        image.save(pred_imgs + '/' + img_name)
    if show:
        image.show()