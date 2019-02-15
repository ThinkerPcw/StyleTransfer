import torch
import torch.nn as nn
import torchvision
from loss.loss import StyleLoss, Content_Loss


class transform_model(nn.Module):
    def __init__(self, style_img, content_img):
        """
        :param style_img:
        :param content_img:
        """
        super(transform_model, self).__init__()
        self.style_img = style_img
        self.content_img = content_img
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_losses = []
        self.style_losses = []
        basenet = torchvision.models.vgg19(pretrained=True).features.cuda()
        self.basenet = self.init_model(basenet)

    def init_model(self, net):
        i = 1
        # we set normalization as a layer in this model
        normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        model = nn.Sequential(normalization)
        for layer in list(net):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                # notice!! The inplace=True in vgg19,This will cause an error in the calculation of the loss function
                model.add_module(name, nn.ReLU(inplace=False))
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                model.add_module(name, layer)

            if isinstance(layer, nn.BatchNorm2d):
                name = "" + str(i)
                model.add_module(name, layer)

            if name in self.content_layers:
                target_feature = model(self.content_img)
                content_loss = Content_Loss(target_feature)
                model.add_module("content_loss_" + str(i), content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(self.style_img)
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_" + str(i), style_loss)
                self.style_losses.append(style_loss)

            if i == 6:
                return model


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).cuda()
        self.std = torch.tensor(std).view(-1, 1, 1).cuda()

    def forward(self, img):
        return (img - self.mean) / self.std
