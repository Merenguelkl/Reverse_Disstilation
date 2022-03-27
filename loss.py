import torch
import torch.nn as nn

def get_ano_map(feature1, feature2):
    mseloss = nn.MSELoss(reduction='none') #1*C*H*W
    mse = mseloss(feature1, feature2) #1*C*H*W
    mse = torch.mean(mse,dim=1) #1*H*W
    cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
    ano_map = torch.ones_like(cos)-cos
    loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
    return ano_map.unsqueeze(1), loss, mse.unsqueeze(1)

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        
    def forward(self, feature1, feature2):
        cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0],-1).mean(-1)).mean()
        return loss


# x1 = torch.rand(2,10,50,50)

# x2 = torch.rand(2,10,50,50)

# cos = CosineLoss()

# print(cos(x1, x2))