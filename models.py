import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchinfo import summary
import torch


# class AttentionPool2D(nn.Module):
#     '''@yewon (ga06033@yonsei.ac.kr)
#         Devised to weight interest region in final layer
#     '''
#     def __init__(self, d_model):
#         super(AttentionPool2D, self).__init__()
#         self.key = nn.Parameter(torch.randn(d_model, 1), requires_grad=True)
#         self.d_model = d_model 


#     def forward(self, x:torch.Tensor):
#         B, C, H, W = x.shape
#         self.key.to(x)
#         x = x.reshape(B, C, -1).permute(0, 2, 1) # (B, N, C)
#         z = (torch.mul(x,self.key.T))/(self.d_model**0.5)     # (B, N, 1)
#         z = torch.softmax(z, dim=1)
#         o = torch.mean(x*z, dim=1)               # (B, C)
#         return o

class AttentionPool2D(nn.Module):
    '''@yewon (ga06033@yonsei.ac.kr)
        Devised to weight interest region in final layer
    '''
    def __init__(self, d_model, dropout=0.1):
        super(AttentionPool2D, self).__init__()
        self.mhsa = nn.MultiheadAttention(d_model, 8, dropout=dropout)
        self.d_model = d_model 


    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)      # (B, N, C)
        x = self.mhsa(x, x, x, need_weights=False)[0] # (B, N, C)
        x = torch.mean(x, dim=1)                      # (B, C)
        return x



class R34_ver1(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_ver1, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-2])])) # drop last layer which is classifier

        ## freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        ## avg pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ## customized classifier layers
        self.classfier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classfier(feat_1d)

        return logit


class R34_attnPool(nn.Module):
    def __init__(self, num_cls=50, freeze_backbone=False):
        super(R34_attnPool, self).__init__()
        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-2])])) # drop last layer which is classifier

        ## freezing backbone parameters
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        ## avg pooling layer
        self.pool = AttentionPool2D(512)

        ## customized classifier layers
        self.classfier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_cls),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.pool(feat_map)
        logit = self.classfier(feat_1d)

        return logit
    
if __name__ == '__main__':
    model = R34_attnPool(50, True)
    summary(model, (1, 3, 224, 224), device='cpu')
