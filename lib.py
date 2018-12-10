import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

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

def get_simple_masked_img(pt_tensor, raw_img, img_name, persist=False, show=False):
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