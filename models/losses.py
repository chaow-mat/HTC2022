# losses 
import torch
# import models.networks_vae_gan as networks
import argparse
import options.options as option

class MultiLoss(torch.nn.Module):
    def name(self):
        return 'MultiLoss'

    def __init__(self, type = 'l1'):
        super(MultiLoss, self).__init__()
        if type == 'l1':
            self.criterionGAN = torch.nn.L1Loss().cuda()
        else:
            self.criterionGAN = torch.nn.MSELoss().cuda()

    # def get_g_loss(self, net, fakeB, realB = None):
    #     # First, G(A) should fake the discriminator
    #     pred_fake = net.forward(fakeB)
    #     return self.criterionGAN(pred_fake, 1)

    def get_loss(self, fakeB, realB):
        self.loss = {}
        num_level = len(fakeB)
        realB_level = []
        for i in range(num_level):
            realB_level.append(torch.nn.functional.interpolate(realB,scale_factor=(0.5)** i))
        realB_level = realB_level[::-1]
        for i in range(num_level):
            self.loss['level_'+str(i)] = self.criterionGAN(fakeB[i],realB_level[i])
        
        return sum(self.loss.values())

    def __call__(self, fakeB, realB):
        return self.get_loss(fakeB, realB)

class DiscLoss(torch.nn.Module):
    def name(self):
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = torch.nn.MSELoss().cuda()

        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, default = '/home/jili/HTC_code/confs/vae_ct_conf.yml', help='Path to option YMAL file.')
        args = parser.parse_args()
        opt = option.parse(args.opt, is_train=False)
        self.net = networks.define_G(opt, step=0).cuda()
        load_net = torch.load('/home/jili_cw4/Deblur_VAE_code/experiments/train_vae_ct/models/59000_G.pth')
        self.net.load_state_dict(load_net, strict=True)
        # set the network is fixed
        for param in self.net.parameters():
            param.requires_grad = False

    def get_g_loss(self, fakeB):
        # First, G(A) should fake the discriminator
        pred_fake = self.net.forward(fakeB)
        return self.criterionGAN(pred_fake, fakeB)

    def get_loss(self, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_enc = self.net.forward(fakeB, 'enc')
        self.real_enc = self.net.forward(realB, 'enc')
        self.loss_D = self.criterionGAN(self.pred_enc,self.real_enc.detach())
        return self.loss_D

    def __call__(self, fakeB, realB):
        return self.get_loss(fakeB, realB)
