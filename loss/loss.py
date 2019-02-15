import torch
from torch import nn


class Content_Loss(nn.Module):
    def __init__(self, target):
        super(Content_Loss, self).__init__()
        # 必须要用detach来分离出target，否则会计算目标值的梯度
        self.target = target.detach()
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.loss = self.criterion(inputs, self.target)
        return inputs


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.target = self.gram(target).detach()
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.G = self.gram(inputs)
        self.loss = self.criterion(self.G, self.target)
        return inputs


class GramMatrix(nn.Module):
    def forward(self, inputs):
        a, b, c, d = inputs.size()  # a=batch size(=1)
        features = inputs.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)
