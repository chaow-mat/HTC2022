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

import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
# import models.networks_vae_gan as networks
# from models.modules.PD_arch import LPD
from models.modules.UNet_arch import U_Net
# from models.modules.MIMO_arch import MIMOUNet
# from models.modules.MSUNet_arch import U_Net
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from .losses import DiscLoss, MultiLoss
from models.modules.operator import OperatorModule, OperatorModuleT

import scipy.io as sio
import numpy as np
logger = logging.getLogger('base')


class LCTModel(BaseModel):
    def __init__(self, opt, step):
        super(LCTModel, self).__init__(opt)
        self.opt = opt
        
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        self.netG = U_Net().to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        if train_opt['loss'] == 'l2':
            self.metric = nn.MSELoss().to(self.device)
        else:
            self.metric = nn.L1Loss().to(self.device)
        # self.metric = MultiLoss(type=train_opt['loss'])

        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            print("WARNING: skipping initial loading, due to resume_state None")

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

        
        # self.regu_loss = DiscLoss()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": self.netG.parameters(), "lr": train_opt['lr_G'],
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G}
            ],
        )

        self.optimizers.append(self.optimizer_G)
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def feed_data(self, data, need_GT=True):
        self.var_L = data['fbp'].to(self.device)  # LQ
        self.sino = data['Sino'].to(self.device)
        if need_GT:
            self.real_H = data['label'].to(self.device)  # GT

    def optimize_parameters(self, step):

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()

        losses = {}
        weight_x = opt_get(self.opt, ['train', 'weight_x'])
        weight_x = 1 if weight_x is None else weight_x
        # x_init = 
        if weight_x > 0:
            self.fake_H = self.netG(self.var_L)
            losses_x = self.metric(self.fake_H,self.real_H)
            losses['x'] = losses_x * weight_x

        weight_y = opt_get(self.opt, ['train', 'weight_y']) or 0
        if weight_y > 0:
            self.fake_sino = self.M(self.fake_H)
            self.sino = self.M(self.real_H).detach()
            losses_y = self.metric(self.fake_sino,self.sino)
            losses['y'] = losses_y * weight_y

        # self.fake_H = self.netG(self.var_L)
        # total_loss = self.metric(self.fake_H,self.real_H)
        # losses['reg_1'] = self.regu_loss.get_g_loss(self.fake_H)
        # losses['reg_2'] = self.regu_loss(self.fake_H, self.real_H)
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer_G.step()

        mean = total_loss.item()
        return mean, losses # print the losses 

    def test(self):
        self.netG.eval()
        self.fake_H = self.netG(self.var_L)
        total_loss = self.metric(self.fake_H,self.real_H)
        self.netG.train()
        return total_loss.item()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['fbp'] = self.var_L[0].detach()[0].float().cpu()
        if isinstance(self.fake_H,list):
            out_dict['fake_H'] = self.fake_H[-1].detach()[0].float().cpu()
        else:
            out_dict['fake_H'] = self.fake_H.detach()[0].float().cpu()
        # out_dict['fake_H'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['label'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=load_submodule)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
