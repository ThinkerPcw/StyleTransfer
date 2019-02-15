from model.transform_model import transform_model
from Data.utils import *
import torch
from torch import nn
import torch.optim as optim

# load data
content_image_path = "./contents/campus.jpg"
style_image_path = './styles/starry-night.jpg'
content_image = get_image(content_image_path, 128).cuda()
style_image = get_image(style_image_path, 128).cuda()

# define model
model = transform_model(style_image, content_image)
net = model.basenet.cuda().eval()

# define loss
style_losses = model.style_losses
content_losses = model.content_losses

# define input
# input_image = torch.randn(content_image.data.size()).cuda()
input_image = content_image.clone().cuda()

# define optimizer
optimizer = optim.LBFGS([input_image.requires_grad_()])

# We use arrays instead of single variables,because use variable will lead a mistake
# (When passing parameters, the array passes the memory address)
step = [0]

print("Start training......")
while step[0] < 300:
    def closure():
        input_image.data.clamp_(0, 1)
        optimizer.zero_grad()
        net(input_image)
        style_score = 0
        content_score = 0
        for loss in style_losses:
            style_score = style_score + 100000 * loss.loss
        for loss in content_losses:
            content_score = content_score + loss.loss
        loss = style_score + content_score
        loss.backward()
        if step[0] % 10 == 0:
            print("step:", step[0], " style_loss:", style_score.data, " content_loss:", content_score.data)
        step[0] += 1
        return loss


    optimizer.step(closure)
input_img=input_image.cpu().data.clamp_(0, 1)
show_image(input_img)
save_image(input_img,content_image_path.split("/")[-1])
print('End of the training')
