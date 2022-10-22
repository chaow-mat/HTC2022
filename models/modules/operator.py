# generate the operator module
from lib2to3.pgen2.token import OP
from turtle import forward
import numpy as np
import torch
import astra

class OperatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,proj_geo,vol,input):
        # vol and geo are used for the experimental setting

        # ctx.save_for_backward(input) # only needed for non-linear operator
        # then at the backward function part , read the input 
        # input, = ctx.saved_tensors
        # grad_input = None
        ctx.proj_geo = proj_geo
        ctx.vol = vol
        out_shape = len(proj_geo['ProjectionAngles']), proj_geo['DetectorCount']
        proj_id = astra.create_projector('cuda',proj_geo,vol)

        vid = astra.data2d.create('-vol',vol,0)
        sid = astra.data2d.create('-sino',proj_geo,0)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectorId']=proj_id
        cfg['ProjectionDataId']=sid
        cfg['VolumeDataId']=vid
        fpid = astra.algorithm.create(cfg)
        # put value in the pointer
        input_arr = input.cpu().detach().numpy()
        # compute the extra_shape
        extra_shape = input_arr.shape[:-2] # 2-D data
        if extra_shape:
            # Multiple inputs: flatten extra axes
            input_arr_flat_extra = input_arr.reshape((-1,)+input_arr.shape[-2:])
            results = []
            for inp in input_arr_flat_extra:
                astra.data2d.store(vid, inp)
                astra.algorithm.run(fpid)
                results.append(astra.data2d.get(sid))
            # restore the correct shape
            result_arr = np.stack(results).astype(np.float32)
            result_arr = result_arr.reshape(extra_shape + out_shape)
        else:
            astra.data2d.store(vid, input_arr)
            astra.algorithm.run(fpid)
            result_arr = astra.data2d.get(sid)
        # convert back to tensor
        tensor = torch.from_numpy(result_arr).cuda()
        astra.projector.delete(proj_id)
        astra.algorithm.delete(fpid)
        astra.data2d.delete(vid)
        astra.data2d.delete(sid)
        return tensor



    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if not ctx.needs_input_grad[2]:
            return None, None, None
        
        proj_geo = ctx.proj_geo
        vol = ctx.vol
        in_shape = vol['GridRowCount'], vol['GridColCount']
        proj_id = astra.create_projector('cuda',proj_geo,vol)

        vid = astra.data2d.create('-vol',vol,0)
        sid = astra.data2d.create('-sino',proj_geo,0)
        cfg = astra.astra_dict('BP_CUDA')
        cfg['ProjectorId']=proj_id
        cfg['ProjectionDataId']=sid
        cfg['ReconstructionDataId']=vid
        bpid = astra.algorithm.create(cfg)

        # Convert tensor to numpy
        grad_output_arr = grad_output.detach().cpu().numpy()
        extra_shape = grad_output_arr.shape[:-2] # 2-D data
        if extra_shape:
            # Multiple inputs: flatten extra axes
            input_arr_flat_extra = grad_output_arr.reshape((-1,)+grad_output_arr.shape[-2:])
            results = []
            for inp in input_arr_flat_extra:
                astra.data2d.store(sid, inp)
                astra.algorithm.run(bpid)
                results.append(astra.data2d.get(vid))
            # restore the correct shape
            result_arr = np.stack(results).astype(np.float32)
            result_arr = result_arr.reshape(extra_shape + in_shape)
        else:
            astra.data2d.store(sid, grad_output_arr)
            astra.algorithm.run(bpid)
            result_arr = astra.data2d.get(vid)
        # convert back to tensor
        grad_input = torch.from_numpy(result_arr).cuda()
        astra.projector.delete(proj_id)
        astra.algorithm.delete(bpid)
        astra.data2d.delete(vid)
        astra.data2d.delete(sid)
        return None, None, grad_input

class OperatorFunctionT(torch.autograd.Function):
    @staticmethod
    def backward(ctx,grad_output):
        # vol and geo are used for the experimental setting

        # ctx.save_for_backward(input) # only needed for non-linear operator
        # then at the backward function part , read the input 
        # input, = ctx.saved_tensors
        # grad_input = None
        grad_input = None
        if not ctx.needs_input_grad[2]:
            return None, None, None
        proj_geo = ctx.proj_geo
        vol = ctx.vol
        in_shape = len(proj_geo['ProjectionAngles']), proj_geo['DetectorCount']
        proj_id = astra.create_projector('cuda',proj_geo,vol)

        vid = astra.data2d.create('-vol',vol,0)
        sid = astra.data2d.create('-sino',proj_geo,0)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectorId']=proj_id
        cfg['ProjectionDataId']=sid
        cfg['VolumeDataId']=vid
        fpid = astra.algorithm.create(cfg)
        # put value in the pointer
        grad_output_arr = grad_output.cpu().detach().numpy()
        # compute the extra_shape
        extra_shape = grad_output_arr.shape[:-2] # 2-D data
        if extra_shape:
            # Multiple inputs: flatten extra axes
            input_arr_flat_extra = grad_output_arr.reshape((-1,)+grad_output_arr.shape[-2:])
            results = []
            for inp in input_arr_flat_extra:
                astra.data2d.store(vid, inp)
                astra.algorithm.run(fpid)
                results.append(astra.data2d.get(sid))
            # restore the correct shape
            result_arr = np.stack(results).astype(np.float32)
            result_arr = result_arr.reshape(extra_shape + in_shape)
        else:
            astra.data2d.store(vid, grad_output_arr)
            astra.algorithm.run(fpid)
            result_arr = astra.data2d.get(sid)
        # convert back to tensor
        grad_input = torch.from_numpy(result_arr).cuda()
        astra.projector.delete(proj_id)
        astra.algorithm.delete(fpid)
        astra.data2d.delete(vid)
        astra.data2d.delete(sid)
        return None, None, grad_input



    @staticmethod
    def forward(ctx, proj_geo,vol,input):
        # grad_input = None
        ctx.proj_geo = proj_geo
        ctx.vol = vol
        # if not ctx.needs_input_grad[0]:
        #     return None, None, None
        
        # proj_geo = ctx.proj_geo
        # vol = ctx.vol
        out_shape = vol['GridRowCount'], vol['GridColCount']
        proj_id = astra.create_projector('cuda',proj_geo,vol)

        vid = astra.data2d.create('-vol',vol,0)
        sid = astra.data2d.create('-sino',proj_geo,0)
        cfg = astra.astra_dict('BP_CUDA')
        cfg['ProjectorId']=proj_id
        cfg['ProjectionDataId']=sid
        cfg['ReconstructionDataId']=vid
        bpid = astra.algorithm.create(cfg)

        # Convert tensor to numpy
        input_arr = input.detach().cpu().numpy()
        extra_shape = input_arr.shape[:-2] # 2-D data
        if extra_shape:
            # Multiple inputs: flatten extra axes
            input_arr_flat_extra = input_arr.reshape((-1,)+input_arr.shape[-2:])
            results = []
            for inp in input_arr_flat_extra:
                astra.data2d.store(sid, inp)
                astra.algorithm.run(bpid)
                results.append(astra.data2d.get(vid))
            # restore the correct shape
            result_arr = np.stack(results).astype(np.float32)
            result_arr = result_arr.reshape(extra_shape + out_shape)
        else:
            astra.data2d.store(sid, input_arr)
            astra.algorithm.run(bpid)
            result_arr = astra.data2d.get(vid)
        # convert back to tensor
        tensor = torch.from_numpy(result_arr).cuda()
        astra.projector.delete(proj_id)
        astra.algorithm.delete(bpid)
        astra.data2d.delete(vid)
        astra.data2d.delete(sid)
        return tensor
class OperatorModule(torch.nn.Module):
    def __init__(self,proj_geo,vol):
        super(OperatorModule, self).__init__()
        self.proj_geo = proj_geo
        self.vol = vol

    def forward(self,x):
        return OperatorFunction.apply(self.proj_geo, self.vol,x)

class OperatorModuleT(torch.nn.Module):
    def __init__(self,proj_geo,vol):
        super(OperatorModuleT, self).__init__()
        self.proj_geo = proj_geo
        self.vol = vol

    def forward(self,x):
        return OperatorFunctionT.apply(self.proj_geo, self.vol,x)

if __name__ == "__main__":
    import os
    import scipy.io as sio
    Data = sio.loadmat('/home/jili/HTC2022/htc2022_ta_full.mat',struct_as_record=False, squeeze_me=True)
    Rec_Data = sio.loadmat('/home/jili/HTC2022/htc2022_ta_full_recon_fbp.mat',struct_as_record=False, squeeze_me=True)
    sinogram1 = Data['CtDataFull'].sinogram
    Data = sio.loadmat('/home/jili/HTC2022/htc2022_tb_full.mat',struct_as_record=False, squeeze_me=True)
    # Rec_Data = sio.loadmat('/home/jili/HTC2022/htc2022_ta_full_recon_fbp.mat',struct_as_record=False, squeeze_me=True)
    sinogram2 = Data['CtDataFull'].sinogram
    ref_sol = Rec_Data['reconFullFbp']
    Rec_Data = sio.loadmat('/home/jili/HTC2022/htc2022_tb_full_recon_fbp.mat',struct_as_record=False, squeeze_me=True)
    ref_sol_2 = Rec_Data['reconFullFbp']
    parameters = Data['CtDataFull'].parameters

    n = 512
    vol = astra.create_vol_geom(n,n)
    # Create shorthands for needed variables
    DSD = parameters.distanceSourceDetector
    DSO = parameters.distanceSourceOrigin
    M  = parameters.geometricMagnification
    angles = parameters.angles
    anglesRad  = np.deg2rad(angles)
    numDetectors = parameters.numDetectorsPost
    effPixel  = parameters.effectivePixelSizePost

    # Distance from origin to detector
    DOD = DSD - DSO

    # Distance from source to origin specified in terms of effective pixel size
    DSO = DSO / effPixel

    # Distance from origin to detector specified in terms of effective pixel size
    DOD = DOD /effPixel

    proj_geo = astra.create_proj_geom('fanflat', M, numDetectors,anglesRad, DSO, DOD)

    # M = OperatorModule(proj_geo,vol)
    # M = M.cuda()
    # inp = torch.from_numpy(ref_sol)[None,None,...].cuda()
    # inp2 = torch.from_numpy(ref_sol_2)[None,None,...].cuda()
    # # out = M(inp)
    # X = torch.cat([inp,inp2],axis=0).type(torch.cuda.FloatTensor)
    # X.requires_grad = True
    # out = M(X)
    # # A = OperatorFunction.apply(proj_geo,vol)
    # # AT = OperatorFunctionT.apply(proj_geo,vol)
    # sino1 = torch.from_numpy(sinogram1)[None,None,...].cuda()
    # sino2 = torch.from_numpy(sinogram2)[None,None,...].cuda()
    # Y = torch.cat([sino1,sino2],axis=0).detach().type(torch.cuda.FloatTensor)
    # mse = torch.nn.MSELoss().cuda()
    # loss = mse(out,Y)
    # loss.backward()
    # print('x')
    # A = OperatorFunction.apply(proj_geo,vol)
    M = OperatorModuleT(proj_geo,vol)
    M = M.cuda()
    # inp = torch.from_numpy(ref_sol)[None,None,...].cuda()
    # inp2 = torch.from_numpy(ref_sol_2)[None,None,...].cuda()
    # out = M(inp)
    sino1 = torch.from_numpy(sinogram1)[None,None,...].cuda()
    sino2 = torch.from_numpy(sinogram2)[None,None,...].cuda()
    X = torch.cat([sino1,sino2],axis=0).type(torch.cuda.FloatTensor)
    X.requires_grad = True
    out = M(X)
    # A = OperatorFunction.apply(proj_geo,vol)
    # AT = OperatorFunctionT.apply(proj_geo,vol)
    # sino1 = torch.from_numpy(sinogram1)[None,None,...].cuda()
    # sino2 = torch.from_numpy(sinogram2)[None,None,...].cuda()
    # Y = torch.cat([sino1,sino2],axis=0).detach().type(torch.cuda.FloatTensor)
    mse = torch.nn.MSELoss().cuda()
    inp = torch.from_numpy(ref_sol)[None,None,...].cuda()
    inp2 = torch.from_numpy(ref_sol_2)[None,None,...].cuda()
    # out = M(inp)
    Y = torch.cat([inp,inp2],axis=0).detach().type(torch.cuda.FloatTensor)
    loss = mse(out,Y)
    loss.backward()
    print('x')