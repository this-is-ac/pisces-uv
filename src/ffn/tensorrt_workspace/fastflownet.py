import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
#from .correlation_package.correlation import Correlation

import torch.onnx.symbolic_opset11 as sym_opset
import torch.onnx.symbolic_helper as sym_help
from torch.onnx import register_custom_op_symbolic
import onnx_graphsurgeon as gs
import onnx
import numpy as np
import tensorrt as trt


def grid_sampler(g, input, grid, mode, padding_mode, align_corners): #long, long, long: contants dtype
    mode_i = sym_help._maybe_get_scalar(mode)
    paddingmode_i = sym_help._maybe_get_scalar(padding_mode)
    aligncorners_i = sym_help._maybe_get_scalar(align_corners)

    return g.op("GridSampler", input, grid, interpolationmode_i=mode_i, paddingmode_i=paddingmode_i,
                aligncorners_i=aligncorners_i) #just a dummy definition for onnx runtime since we don't need onnx inference
sym_opset.grid_sampler = grid_sampler
register_custom_op_symbolic("::grid_sampler",grid_sampler,11)

def Correlation(g,input1,input2,kH,kW,patchH,patchW,padH,padW,dilationH,dilationW,dilation_patchH,dilation_patchW,dH,dW):
    patchSize_i = sym_help._maybe_get_scalar(patchH)
    dilation_i = sym_help._maybe_get_scalar(dilation_patchH)
    return g.op('Correlation',input1,input2,patchsize_i=patchSize_i,dilation_i=dilation_i)
sym_opset.Correlation = Correlation
register_custom_op_symbolic("mynamespace::correlation",Correlation,11)

script_root = os.path.dirname(__file__)
from glob import glob
ops_so = glob(os.path.join(script_root,'correlation_pytorch/build/*/*.so'))[0]
torch.ops.load_library(ops_so)

def correlation(input1,input2):
    out = torch.ops.mynamespace.correlation(
        input1,input2,\
        1,1,\
        9,9,\
        0,0,\
        0,0,\
        1,1,\
        1,1
    ) / 81.
    return out

def modify_onnx(onnx_model_file,onnx_values_file):

    valuses_lines = open(onnx_values_file).readlines()
    import re
    index2shape = {}
    for line in valuses_lines:
        searchObj = re.search( r'\%\d* ', line, re.M|re.I) #把标号抓出来
        if searchObj is None:
            continue
        id = searchObj[0][:-1]
        if id is None:
            continue
        searchObj = re.search( r'\([\d*, ]+', line, re.M|re.I) #把shape抓出来
        if searchObj is None:
            continue
        shape = eval(searchObj[0][:-2]+')')
        index2shape[id] = shape
    print(index2shape)

    graph = gs.import_onnx(onnx.load(onnx_model_file))
    assert(graph is not None)

    for node in graph.nodes:
        if node.op == 'Correlation': #修改correlation

            inputs_0_id,inputs_1_id = '%'+node.inputs[0].name,'%'+node.inputs[1].name
            outputs_0_id = '%'+node.outputs[0].name

            node.inputs[0].shape = index2shape[inputs_0_id] #先根据记录的标号 把shape给搞出来
            node.inputs[1].shape = index2shape[inputs_1_id]
            node.outputs[0].shape = index2shape[outputs_0_id]

            node.inputs[0].dtype = node.inputs[1].dtype = node.outputs[0].dtype = np.float32

            _, c, h, w = node.inputs[0].shape
            patchSize = node.attrs['patchsize']
            dilation = node.attrs['dilation']

            m_type = 0 if node.inputs[0].dtype == np.float32 else 1

            buffer = np.array([c,h,w,patchSize**2,patchSize,patchSize,dilation],dtype=np.int64).tobytes('C') \
                     + np.array([m_type], dtype=np.int32).tobytes('C')
            node.attrs = {'name':'Correlation', 'version':'1', 'namespace':"", 'data':buffer}
            node.op = 'TRT_PluginV2'

        elif node.op == 'GridSampler': #修改grid_sample

            inputs_0_id,inputs_1_id = '%'+node.inputs[0].name,'%'+node.inputs[1].name
            outputs_0_id = '%'+node.outputs[0].name

            node.inputs[0].shape = index2shape[inputs_0_id]
            node.inputs[1].shape = index2shape[inputs_1_id]
            node.outputs[0].shape = index2shape[outputs_0_id]

            node.inputs[0].dtype = node.inputs[1].dtype = node.outputs[0].dtype = np.float16

            _, c, h, w = node.inputs[0].shape
            _, h_g, w_g, _ = node.inputs[1].shape
            align_corners = node.attrs['aligncorners']
            inter_mode = node.attrs['interpolationmode']
            pad_mode = node.attrs['paddingmode']
            m_type = 0 if node.inputs[0].dtype == np.float32 else 1
            buffer = np.array([c, h, w, h_g, w_g], dtype=np.int64).tobytes('C') \
                     + np.array([inter_mode, pad_mode], dtype=np.int32).tobytes('C') \
                     + np.array([align_corners], dtype=np.bool).tobytes('C') \
                     + np.array([m_type], dtype=np.int32).tobytes('C')
            node.attrs = {'name':'GridSampler', 'version':'1', 'namespace':"", 'data':buffer}
            node.op = 'TRT_PluginV2'

    onnx.save(gs.export_onnx(graph), onnx_model_file)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out


class FastFlowNet(nn.Module):
    def __init__(self, groups=3):
        super(FastFlowNet, self).__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        #correlation = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.index = torch.tensor([0, 2, 4, 6, 8,
                                   10, 12, 14, 16,
                                   18, 20, 21, 22, 23, 24, 26,
                                   28, 29, 30, 31, 32, 33, 34,
                                   36, 38, 39, 40, 41, 42, 44,
                                   46, 47, 48, 49, 50, 51, 52,
                                   54, 56, 57, 58, 59, 60, 62,
                                   64, 66, 68, 70,
                                   72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        xx_plus = xx + torch.index_select(flo,dim=1,index=torch.tensor([0],dtype=torch.long))
        yy_plus = yy + torch.index_select(flo,dim=1,index=torch.tensor([1],dtype=torch.long))
        
        xx_plus = 2.0 * xx_plus / max(W-1,1) - 1.
        yy_plus = 2.0 * yy_plus / max(H-1,1) - 1.
        vgrid = torch.cat([xx_plus,yy_plus],1).to(x)

        # slice in left if not permitted in tensorrt
        # grid = torch.cat([xx, yy], 1).to(x)
        # vgrid = grid + flo
        # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear')
        return output


    def forward(self, x):
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:6, :, :]
        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        corr_out = correlation(f16, f26)
        cv6 = torch.index_select(correlation(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up*0.625)
        cv5 = torch.index_select(correlation(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up


        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up*1.25)
        cv4 = torch.index_select(correlation(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up\

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up*2.5)
        cv3 = torch.index_select(correlation(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up*5.0)
        cv2 = torch.index_select(correlation(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        return flow2

        # if self.training:
        #     return flow2, flow3, flow4, flow5, flow6
        # else:
        #     return flow2

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

class testnet(nn.Module):
    def __init__(self):
        super(testnet, self).__init__()
        self.conv = nn.Conv2d(6,6,3,1,1)
    def forward(self,data):
        return self.conv(data)

if __name__ == '__main__':

    flownet = FastFlowNet().eval()
    flownet.load_state_dict(torch.load(os.path.join(script_root,'../checkpoints/fastflownet_ft_mix.pth')))

    #flownet = testnet().eval()
    import cv2
    from flow_vis import flow_to_color
    img_paths = [os.path.join(script_root,'../data/img_050.jpg'),\
                 os.path.join(script_root,'../data/img_051.jpg')]

    div_flow = 20.0
    div_size = 64
    fig1,fig2 = cv2.imread(img_paths[0]),cv2.imread(img_paths[1])
    fig1,fig2 = cv2.resize(fig1,(512,512)),cv2.resize(fig2,(512,512))

    img1 = torch.from_numpy(fig1).float().permute(2,0,1)[None] / 255.
    img2 = torch.from_numpy(fig2).float().permute(2,0,1)[None] / 255.
    img1,img2,_ = centralize(img1,img2)
    input_t = torch.cat([img1,img2],1)
    with torch.no_grad():
        output = flownet(input_t)
    pytorch_output = output.squeeze().detach().numpy().squeeze()
    #
    flow = div_flow * output
    flow = flow[0].cpu().permute(1,2,0).numpy()
    flow_color = flow_to_color(flow,convert_to_bgr=True)

    cv2.namedWindow('pytorch flow',cv2.WINDOW_NORMAL)
    cv2.imshow('pytorch flow',flow_color)
    # cv2.waitKey(0)

    #expoert onnx
    onnx_f = "flownet.onnx"
    onnx_values_f = "flownet.onnx.values"

    onnx_values = open(onnx_values_f,'w')
    sys.stdout.write_bak = sys.stdout.write
    sys.stdout.write = onnx_values.write
    #enable_onnx_checker is deprecated, so we can not control exporter check the onnx file, but onnx file will be generated anyway
    try:
        torch.onnx.export(flownet,(input_t.float()),onnx_f,verbose=True,\
                          input_names=['input'],output_names=['output'],\
                          opset_version=11,export_params=True,enable_onnx_checker=False)
    except:
        pass
    sys.stdout.write = sys.stdout.write_bak
    modify_onnx(onnx_f,onnx_values_f)

    from common import *
    import ctypes
    ctypes.CDLL(open(os.path.join(os.path.dirname(__file__),'tensorrt_plugin_path')).read())
    import pycuda.driver as cuda
    import pycuda.autoinit
    with build_engine_onnx(onnx_f,fp16=True) as engine, open('engine_fp16','wb') as f:
        f.write(engine.serialize())
        inputs,outputs,bindings,stream = allocate_buffers(engine,True,2)
        for binding in engine:
            print('-------------------')
            print(engine.get_binding_shape(binding))
            print(engine.get_binding_name(engine.get_binding_index(binding)))
        with engine.create_execution_context() as context:
            input_t = input_t.float().numpy()
            inputs[0].host = input_t
            trt_outputs = do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
            output = trt_outputs[0].reshape(pytorch_output.shape)

            flow = div_flow * output
            flow = np.transpose(flow,[1,2,0])
            flow_color = flow_to_color(flow,convert_to_bgr=True)

            cv2.namedWindow('tensorrt flow',cv2.WINDOW_NORMAL)
            cv2.imshow('tensorrt flow',flow_color)
            cv2.waitKey(0)
