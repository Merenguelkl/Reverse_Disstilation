import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2


class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, stride):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=stride, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(inplace=True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X    

class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(ConvTransposeBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.ConvTranspose2d(in_channel,F1,kernel_size=2,stride=2, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.ConvTranspose2d(in_channel,F3,kernel_size=2,stride=2, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(inplace=True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X    
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.wRes50 = wide_resnet50_2(pretrained=True)

        
    def forward(self, x):
        x = self.wRes50.conv1(x)
        x = self.wRes50.bn1(x)
        x = self.wRes50.relu(x)
        x = self.wRes50.maxpool(x)

        x = self.wRes50.layer1(x) # [1, 256, 64, 64]
        feature1 = x
        
        x = self.wRes50.layer2(x) # [1, 512, 32, 32]
        feature2 = x
        
        x = self.wRes50.layer3(x) # [1, 1024, 16, 16]
        feature3 = x
        
        return feature1, feature2, feature3 

class OCBE(nn.Module):
    def __init__(self):
        super(OCBE, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(inplace=True)
                                     )
        
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU(inplace=True)
                                     )
        
        self.merge =nn.Sequential(nn.Conv2d(in_channels = 3072, out_channels = 1024, kernel_size = 1, stride = 1, padding = 0),
                                  nn.BatchNorm2d(1024),
                                  nn.ReLU(inplace=True)
                                  )
        
        self.resblock = nn.Sequential(ConvBlock(in_channel =1024, kernel_size = 3, filters=[512,512,2048], stride=2),
                                      IndentityBlock(in_channel=2048, kernel_size=3, filters=[512,512,2048]),
                                      IndentityBlock(in_channel=2048, kernel_size=3, filters=[512,512,2048])
                                      )
        
    def forward(self, x1, x2, x3):
        output = torch.cat((self.branch1(x1),self.branch2(x2),x3),dim=1) # [1, 3072, 16, 16]
        output = self.merge(output) # [1, 1024, 16, 16]
        # output = self.branch1(x1) + self.branch2(x2) + x3
        output = self.resblock(output) # [1, 2048, 8, 8]
        
        return output
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer3 = nn.Sequential(ConvTransposeBlock(in_channel=2048, kernel_size=3, filters=[512, 1024, 1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    IndentityBlock(in_channel=1024, kernel_size=3, filters=[512,1024,1024]),
                                    )
        self.layer2 = nn.Sequential(ConvTransposeBlock(in_channel=1024, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512]),
                                    IndentityBlock(in_channel=512, kernel_size=3, filters=[256, 512, 512])
                                    )
        self.layer1 = nn.Sequential(ConvTransposeBlock(in_channel=512, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256]),
                                    IndentityBlock(in_channel=256, kernel_size=3, filters=[128, 256, 256])
                                    )

        
    def forward(self, x):
        x = self.layer3(x) # [1, 1024, 14, 14]
        feature3 = x
        x = self.layer2(x) # [1, 512, 28, 28]
        feature2 = x
        x = self.layer1(x) # [1, 256, 56, 56]
        feature1 = x
        
        return feature1, feature2, feature3

class OcbeAndDecoder(nn.Module):
    def __init__(self):
        super(OcbeAndDecoder, self).__init__()
        self.ocbe = OCBE()
        self.decoder = Decoder()
    def forward(self, e_feature1, e_feature2, e_feature3):
        x = self.ocbe(e_feature1, e_feature2, e_feature3)
        feature1, feature2, feature3 = self.decoder(x)
        return feature1, feature2, feature3
