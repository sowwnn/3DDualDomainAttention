from monai.networks.blocks import Convolution
import torch.nn as nn
import torch

class TMB(nn.Module):
    def __init__(self, in_c, mid_c):
        super(TMB,self).__init__()
        
        self.prj1 = nn.Conv3d(sum(in_c), mid_c, 3, 1, 1)
        self.prj2 = nn.Conv3d(mid_c, mid_c, 1, 1, 0)
        self.prj3 = nn.Conv3d(mid_c, sum(in_c), 3, 1, 1)
        
        
    def forward(self, x):
        
        x_ = self.prj1(x)
        x_ = self.prj2(x_)
        x_ = self.prj3(x_)
        
        
        x_ = torch.mean(torch.stack([x,x_]),0)
        return x_
    
    
class GB(nn.Module):
    def __init__(self, num_heads = 8):
        super(GB, self).__init__()
        
        self.dk = num_heads ** -0.5
        
    def forward(self, q, k, v):
        
        attn = (q @ k) * self.dk
        attn = nn.Softmax(dim=1)(attn)
        x = (attn @ v)
        x = nn.Softmax(dim=1)(x)
        x = (x * v) 
        return x + v

        
        
class LB(nn.Module):
    def __init__(self, in_c, mid_c, out_c, drop=0.3):
        super(LB, self).__init__()
        
        self.conv1 = nn.Conv3d(in_c, mid_c, 1, 1)
        self.TC = nn.Conv3d(mid_c, mid_c, 3, 1, 1)
        self.conv2 = nn.Conv3d(mid_c, out_c, 1,1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        
        
    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.TC(x1)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x1 = self.conv2(x1)
        x1 = self.drop(x1)
        
        return x1 + x
    

class MLP(torch.nn.Module):
    def __init__(self,in_c):
        super(MLP, self).__init__()
        
        self.conv1 =Convolution(
        spatial_dims=3,
        in_channels=in_c,
        out_channels=in_c,
        strides=1, kernel_size=1,
        adn_ordering="ADN",
            act=("relu"),
            dropout=0.1,
            norm=("batch"),
        )
        
        self.conv2 =Convolution(
        spatial_dims=3,
        in_channels=in_c,
        out_channels=3,
        strides=1, kernel_size=1,
        adn_ordering="ADN",
            act=("relu"),
            dropout=0.1,
            norm=("batch"),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Fusion(nn.Module):
    def __init__(self, in_cs, mid_c, out_c):
        super(Fusion, self).__init__()

        self.in_c1 = in_cs[0]
        
        self.TMB = TMB(in_cs ,mid_c)
        
        self.GB = GB()
        
        self.LB = LB(sum(in_cs), mid_c, out_c)
        
        self.TC1 = nn.Conv3d(in_cs[0], mid_c, 1, 1)
        self.TC2 = nn.Conv3d(in_cs[1], mid_c, 1, 1)

        self.mlp = MLP(48)
        
    def forward(self, v):
        
        f1 = v[:,:self.in_c1,:,:,:]
        f2 = v[:,self.in_c1:,:,:,:]
        
        v = torch.cat([f1,f2],1)
        v = self.TMB(v)
        
        f1 = self.TC1(f1)
        f2 = self.TC2(f2)
        
        v = self.GB(f1, f2, v)
        v = self.LB(v) + v
        v = self.mlp(v)       
        return v