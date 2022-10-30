import os
import torch
import numpy as np
from PIL import Image as Image
from torch import angle
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import astra
import scipy.io as sio
import glob
import imutils

class CTDataset(Dataset):
    def __init__(self, image_dir, angle = None, transform=None, is_test=False):
        self.sino_dir = image_dir 
        self.no_angle = angle
        self.sino_list = glob.glob(os.path.join(self.sino_dir,'*.mat'))
        self.sino_list.sort()
        self.sino_data = []
        self.start_angle = []
        # self.parameters = []
        # self.label_data = []
        # read all data into memory
        for file in self.sino_list:
            Data = sio.loadmat(file,struct_as_record=False, squeeze_me=True)
            
            self.sino_data.append(Data['CtDataLimited'].sinogram)
            self.start_angle.append(Data['CtDataLimited'].parameters.angles[0])
            # name = os.path.splitext(file)[0]
            # Rec_Data = sio.loadmat(name + '_recon_fbp.mat',struct_as_record=False, squeeze_me=True)
            # self.label_data.append(Rec_Data['reconFullFbp'])
        self.parameters = Data['CtDataLimited'].parameters
        self.transform = transform
        self.is_test = is_test
        self._define_set()
        template = sio.loadmat('template/htc2022_solid_disc_full_recon_fbp.mat')['reconFullFbp']
        template = np.float32(template)
        self.template = torch.from_numpy(template[None,...])
        

    def __len__(self):
        return len(self.sino_list)

    def __getitem__(self, idx):
        # self._define_set(idx = idx)
        image = self.sino_data[idx]
        # label = self.label_data[idx]

        alpha_start = self.start_angle[idx]
        # else:
        #     alpha_start = np.random.randint(0,(360-self.angle))
        roll_shift = int(alpha_start*2)
        sinogram_trans_full = np.roll(image, -roll_shift, axis=0)
        sinogram_trans = sinogram_trans_full[:len(self.anglesRad),:]

        astra.data2d.store(self.sid,sinogram_trans)
        astra.algorithm.run(self.fbp)
        fbp = astra.data2d.get(self.vid)
        fbp = np.float32(fbp)

        name = os.path.splitext(os.path.basename(self.sino_list[idx]))[0]
        # label = imutils.rotate(label,angle=-alpha_start)
        if self.is_test:
            # transform the images
            # fbp = imutils.rotate(fbp,angle=alpha_start) # no need to rotate
            # print(fbp.shape)
            fbp = F.to_tensor(fbp)
            sinogram_trans = F.to_tensor(sinogram_trans)
            # label = F.to_tensor(label)
            # label = imutils.rotate(label,angle=-alpha_start)
        return {'Sino': sinogram_trans, 'fbp': torch.cat([fbp, self.template],axis=0),'name':name,'alpha_start': alpha_start} 

    def _define_set(self):
        n = 512
        vol = astra.create_vol_geom(n,n)
        # Create shorthands for needed variables
        DSD = self.parameters.distanceSourceDetector
        DSO = self.parameters.distanceSourceOrigin
        M  = self.parameters.geometricMagnification
        if not self.no_angle:
            self.no_angle = len(self.parameters.angles)
        # self.start_angle = self.parameters.angles[0]
        angles = np.arange(0, self.no_angle+0.5,0.5)
        self.anglesRad = np.deg2rad(angles)

        numDetectors = self.parameters.numDetectorsPost
        effPixel  = self.parameters.effectivePixelSizePost

        # Distance from origin to detector
        DOD = DSD - DSO

        # Distance from source to origin specified in terms of effective pixel size
        DSO = DSO / effPixel

        # Distance from origin to detector specified in terms of effective pixel size
        DOD = DOD /effPixel

        proj_geo = astra.create_proj_geom('fanflat', M, numDetectors,self.anglesRad, DSO, DOD)
        proj_id = astra.create_projector('cuda',proj_geo,vol)
        self.vid = astra.data2d.create('-vol',vol,0)
        self.sid = astra.data2d.create('-sino',proj_geo,0)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectionDataId']=self.sid
        cfg['ReconstructionDataId']=self.vid
        cfg['ProjectorID'] = proj_id
        # cfg['option'] = {'FilterType':'Ram-Lak'}
        self.fbp = astra.algorithm.create(cfg)
        # print(len(anglesRad))
        # print(len(anglesRad))

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
    
if __name__ == "__main__":
    # create the training dataset for the limited-angle CT
    import h5py
    import os
    ANGLES = [20,10] #[180] #[180, 90] #[180, 90, 80, 70, 60, 50, 40, 30]
    Save_path = 'data/train_add/'
    alphas = np.arange(0, 360+0.5,0.5)
    
    
    for angle in ANGLES:
        # open a h5py file
        
        dataset = CTDataset(image_dir='/home/jili/HTC2022/', angle=angle)
        for i, file in enumerate(dataset.sino_list):
            file_s = os.path.basename(file)
            name = os.path.splitext(file_s)[0]
            print(name)
            if not os.path.exists(os.path.join(Save_path,'Ang_'+ str(angle))):
                os.makedirs(os.path.join(Save_path,'Ang_'+ str(angle)),exist_ok=True)
            h5file = h5py.File(os.path.join(Save_path,'Ang_'+ str(angle),name +'.h5'),'w')
            Sino_data = []
            fbp_data = []
            label_data = []
            for alpha in alphas:
                Out_dict = dataset.__getitem__(i, alpha_start=alpha)
                Sino_data.append(Out_dict['Sino'])
                fbp_data.append(Out_dict['fbp'])
                label_data.append(Out_dict['label'])
            h5file.create_dataset('Sino',data=np.stack(Sino_data,0))
            h5file.create_dataset('fbp',data = np.stack(fbp_data,0))
            h5file.create_dataset('label',data = np.stack(label_data,0))
            h5file.close()
