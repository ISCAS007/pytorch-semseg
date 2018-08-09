# -*- coding: utf-8 -*-

import os
from easydict import EasyDict as edict
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
#from torch.autograd import Variable

from ptsemseg.loader import get_loader, get_data_path

def get_dataset_loader():
    args=edict()
    args.dataset_name = 'cityscapes'
    args.config_path = os.path.join('/home/yzbx/git/gnu/pytorch-semseg', 'config.json')
    args.img_rows=224
    args.img_cols=224
    args.img_norm=True
    args.batch_size=2
    
    
    data_loader = get_loader(args.dataset_name)
    data_path = get_data_path(args.dataset_name, args.config_path)
    t_loader = data_loader(data_path, is_transform=True, split='train', img_size=(
        args.img_rows, args.img_cols), augmentations=None, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(
        args.img_rows, args.img_cols), img_norm=args.img_norm)
    
    n_classes = t_loader.n_classes
    print('class number is',n_classes)
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=8)
    
    return trainloader,valloader

if __name__ == '__main__':
    trainloader,_=get_dataset_loader()
    for i, (images, labels) in enumerate(trainloader):
        print(images.shape)
        print(labels.shape)
    #    images_val = Variable(images.cuda(), volatile=True)
    #    labels_val = Variable(labels.cuda(), volatile=True)
    
        images_np = images.data.cpu().numpy()
        labels_np = labels.data.cpu().numpy()
        
        print(images_np.shape)
        print(labels_np.shape)
        print(np.unique(labels_np))
        plt.imshow(labels_np[0,:,:])
        plt.show()
        break