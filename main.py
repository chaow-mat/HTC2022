# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE


from models import create_model
import torch
import numpy as np
# import pandas as pd
import os
import logging
import numpy as np
from skimage.io import imsave

import gdown

import imutils
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing
import yaml
from utils.util import OrderedYaml

Loader, Dumper = OrderedYaml()

class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def from_gdrive_download(save_path = None, url=None, checkpoint_name=None):
    output = checkpoint_name 
    gdown.download(url, os.path.join(save_path, output),quiet=False)
    print('Download is okay!')

def seg(img):
    img[img<0] = 0.0
    level = threshold_otsu(img)
    img[img> level] = 1.0
    img[img<= level] = 0.0
    img = area_closing(img,area_threshold=3)
    return img

def main():
    import argparse

    torch.set_num_threads(3)

    parser = argparse.ArgumentParser() # deleter --
    parser.add_argument('input_path', type=str, default='./HTC2022/', help='the path of sinogram .mat data')
    parser.add_argument('output_path', type=str, default='./submit/HDC2022_result/', help='output path of the recovery binary images')
    parser.add_argument('n_cat', type=int, default=3, help='the level of input')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='1', help='use gpu or cpu')
    args = parser.parse_args()

    args.gpu_id = args.gpu_id if torch.cuda.is_available() else '-1'
    args.n_cat = str(args.n_cat)
    if int(args.gpu_id) >=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.makedirs(args.output_path, exist_ok=True)
    logging.basicConfig(filename= args.output_path + 'runtime.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    angle = {'1':90,
            '2':80,
            '3':70,
            '4':60,
            '5':50,
            '6':40,
            '7':30}[args.n_cat]
    # define the code
    checkpoint_name = {'1':'htc2022_n_cat_1.pth',
                        '2':'htc2022_n_cat_2.pth',
                        '3':'htc2022_n_cat_3.pth',
                        '4':'htc2022_n_cat_4.pth',
                        '5':'htc2022_n_cat_5.pth',
                        '6':'htc2022_n_cat_6.pth',
                        '7':'htc2022_n_cat_7.pth',
                        }[args.n_cat]

    url = {'1':'https://drive.google.com/uc?id=1kGA9zUTsnrkR80zE8ZJjuPsaPCF40bCr',
            '2':'https://drive.google.com/uc?id=11MDOyGghu6gwR3l64HSap_waZNJzct9E',
            '3':'https://drive.google.com/uc?id=1stN8HfiWC52K9S-lcuq9yj1rIuJZClA9',
            '4':'https://drive.google.com/uc?id=1eE1fOJZE0M7Ar6TthVWqfYGrAkqXbwdX',
            '5':'https://drive.google.com/uc?id=1qiG3TnR7v-rvLJJnMIP_XaI26S8UetwT',
            '6':'https://drive.google.com/uc?id=1kOak1fENLfkCQmYP7C6dTz4iNRsqBnCX',
            '7':'https://drive.google.com/uc?id=1eCB7edWTilzMzhIoEX_fz6foL4hKU8xy'
            }[args.n_cat]
    # to check the existences, wether 
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints',exist_ok=True)
    
    if not os.path.exists(os.path.join('checkpoints',checkpoint_name)):
        from_gdrive_download(save_path='checkpoints',url=url, checkpoint_name=checkpoint_name)
    device = 'cuda' if int(args.gpu_id) >=0 else 'cpu'
    model_path = './checkpoints/' + checkpoint_name
    
    opt_path = './confs/vae_ct_conf.yml'
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    
    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)
    model = create_model(opt)

    print(model_path)
    model.load_network(load_path=model_path, network=model.netG)
    model.netG.eval()

    from data.limit_angle_dataset import CTDataset as D
    dataset = D(args.input_path, angle=angle, is_test=True)
    
    img_dir = args.output_path
    if not os.path.exists(img_dir):
        os.makedirs(img_dir,exist_ok=True)

    idx = 0
    for ik in range(len(dataset)):
        idx += 1
        val_data = dataset.__getitem__(ik)
        img_name = val_data['name']
        val_data['fbp'] = val_data['fbp'][None,...]
        val_data['Sino'] = val_data['Sino'][None,...]

        model.feed_data(val_data,need_GT=False)

        with torch.no_grad():
            fake_H = model.netG(val_data['fbp'])

        x = fake_H.cpu().numpy().squeeze()
        rec = (x - np.min(x))/(np.max(x)-np.min(x))
        
        rec = imutils.rotate(rec,angle=val_data['alpha_start']) #
        save_img_path = os.path.join(img_dir,'{:s}.png'.format(img_name))
        imsave(save_img_path,seg(rec))

if __name__ == "__main__":
    main()
