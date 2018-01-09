import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import random
import numpy as np


class ReadImage(Dataset):

    # set the random seed.
    # actions np array.
    # rewards np.array.

    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    mean_bgr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    stand_bgr = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, imgs_list, actions, rewards, transform=None):

        super(ReadImage,self).__init__()

        # TODO
        # For different images, we need to resize the images to 224*224
        # or padding, without any extra information.

        self.new_size = (224, 224, 3)
        self.len= len(imgs_list)
        self.images = imgs_list
        self.actions = actions
        self.rewards = rewards
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        imgs = self.images[idx]

        action = self.actions[idx]
        reward = self.rewards[idx]
        # print('raw_iamge', imgs.shape)
        # resize the image
        img = np.resize(imgs, self.new_size)
        # print('img_size', img.shape)
        img = np.array(img, dtype=np.float64)

        img_np = img[:, :, ::-1]
        img_np = img_np/255.0 - self.mean_bgr
        img_np /= self.stand_bgr

        sample = {'images':img_np, 'actions': action, 'rewards': reward}

        # transform the format
        if self.transform:
            sample = self.transform(sample)
        return sample


class Numpy2Tensor(object):

    """converting the numpy format to """
    def __call__(self, sample):

        image, action, reward = sample['images'], sample['actions'], sample['rewards']
        # print('before_transpose:', image.shape)
        image = image.transpose(2, 0, 1)
        action = np.array([action])
        reward = np.array([reward])

        image = torch.from_numpy(image).float()
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(reward).float()
        # print ('images_size', image.size())

        return {'images':image, 'actions': action, 'rewards': reward}


def get_load(imgs, actions, rewards, batch_size=32):

    train_dataset = ReadImage(imgs_list=imgs,
                              actions=actions,
                              rewards=rewards,
                              transform=transforms.Compose([Numpy2Tensor()]))

    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             num_workers=10,
                             shuffle=True)
    return trainloader


def ReadSingleImage(img):
    """
    Read single images,
    Args:
    ----
    - img:  np.array format, h*w*c
    - img: return, torch.FloatTensor
    """

    new_size = (128, 128, 3)
    # new_size = (224,224, 3)

    mean_bgr = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    stand_bgr = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.resize(img, new_size)
    # img = img.resize(new_size)
    img = img[:, :, ::-1]
    img = img/255.0 - mean_bgr
    img /= stand_bgr
    img = img.transpose(2, 0, 1)
    img = img[None, :, :, :] # 1*c*h*w
    img = torch.from_numpy(img).float()
    return img


if __name__ == '__main__':

    import logging

    # for idx, sample in enumerate(train_loader):
    #     image, label = sample['image'], sample['label']
    #     image_name = sample['image_name']
    #     print image_name, label
