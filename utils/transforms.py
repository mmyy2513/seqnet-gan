from __future__ import print_function
import random

import sys
dgnet_path = "/root/workplace/PersonSearch/DG-Net"
sys.path.append(dgnet_path)
import PIL

sys.path.append('.')

#print(sys.path)

from util import get_config
from trainer import DGNet_Trainer, to_gray

import argparse
from torch.autograd import Variable

import torch
import os
import numpy as np
from torchvision import datasets
from PIL import Image
import cv2

from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms




def recover2(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    #inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp



# torch.manual_seed(10)
# torch.cuda.manual_seed(10)
if not os.path.exists("res/"):
    os.makedirs("res/")
    

#print("!done")
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target
    
class color2gray:
    def __call__(self, image, target):
        image = transforms.Grayscale(num_output_channels=3)(image)
        return image, target
    

class ToTensor:
    def __call__(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)
        return image, target
"""
class RandomCloth:
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        save_name = target['img_name']
        
        if random.random() < self.prob:
            im = np.array(image)
            
            origin = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cv2.imwrite("res/original/"+save_name,origin)   
            
            coords = target['boxes']
            
            gallery = []
            size_origin = []
            coord_origin = []
            
            for n, i in enumerate(coords):
                x1, y1, x2, y2 = int(i[0].item()), int(i[1].item()), int(i[2].item()), int(i[3].item())
                size_origin.append((abs(x1-x2), abs(y1-y2)))
                coord_origin.append((x1,y1,x2,y2))
                warp = cv2.resize(im[y1:y2, x1:x2], (128,256), interpolation= 3)
                #cv2.imwrite("warp"+save_name, warp)
                warp = transforms.ToTensor()(warp).unsqueeze(0)
                warp = tf(warp)
                #cv2.imwrite("warpp+"+save_name,recover2(warp.squeeze(0).data.cpu()))
                gallery.append(warp)
                
            if len(gallery) > 1:
                trainer = DGNet_Trainer(config)

                state_dict_gen = torch.load(dgnet_path+"/outputs/%s/checkpoints/gen_00100000.pt"%name)
                trainer.gen_a.load_state_dict(state_dict_gen['a'], strict=False)
                trainer.gen_b = trainer.gen_a

                state_dict_id = torch.load(dgnet_path+"/outputs/%s/checkpoints/id_00100000.pt"%name)
                trainer.id_a.load_state_dict(state_dict_id['a'])
                trainer.id_b = trainer.id_a
                
                trainer.to(device)
                trainer.eval()
                encode = trainer.gen_a.encode # encode function
                style_encode = trainer.gen_a.encode # encode function
                id_encode = trainer.id_a # encode function
                decode = trainer.gen_a.decode # decode function

                ind = random.randint(0, len(gallery))
                #print(ind)
                structure = torch.stack(gallery).squeeze(1)
                #print(gallery[ind])
                bg_img = structure
                gray = to_gray(False)
                bg_img = gray(bg_img)
                bg_img = Variable(bg_img.to(device))
                
                id_img = gallery[ind]
                
                cv2.imwrite("id"+save_name,recover(id_img.squeeze(0).data.cpu()))
                id_img = Variable(id_img.to(device))
                n, c, h, w = id_img.size()
                
                s = encode(bg_img)
                
                #print(id_img.shape)
                #cv2.imwrite("id"+save_name,recover(id_img.squeeze(0).data.cpu()))
                f, _ = id_encode(id_img)
                
                out = []
                for i in range(s.size(0)):
                    s_tmp = s[i,:,:,:]
                    outputs = decode(s_tmp.unsqueeze(0), f)
                    tmp = recover(outputs[0].data.cpu())
                    out.append(tmp)

                for i in range(len(out)):
                    out[i] = cv2.resize(out[i], (size_origin[i][0], size_origin[i][1]))
                    im[coord_origin[i][1]:coord_origin[i][3], coord_origin[i][0]:coord_origin[i][2]] = out[i]
                    
                tmp3 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                cv2.imwrite("res/after_"+save_name, tmp3)

            return im, target   
            
        return image, target
"""       

def build_transforms(is_train):
    transforms = []
    
    if is_train:
        #transforms.append(RandomCloth())
        transforms.append(ToTensor())
        transforms.append(RandomHorizontalFlip())
    else:
        transforms.append(ToTensor())
    return Compose(transforms)
 