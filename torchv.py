'''
函数说明:  这里也和老师的不一样
Author: hongqing
Date: 2021-09-03 11:31:51
LastEditTime: 2021-09-16 16:55:04
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torchvision import models
%matplotlib inline
np.random.seed(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader,Dataset
from torchvision import transforms
import copy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import cv2
style_img_path = r'images\你自己的风格图片.jpg'
content_img_path = r'images\你自己的内容图片.png'
height,width,_ = cv2.imread(style_img_path).shape
## 输入你们自己处理一下啊 GPU加载不动就把这个参数改大点 图片就变小了
cutsize = 8
height = height//cutsize
width = width//cutsize

loader = transforms.Compose([transforms.ToTensor(),transforms.Resize([height,width])])

style_img = loader(cv2.imread(style_img_path)).unsqueeze(0).to(device)
content_img = loader(cv2.imread(content_img_path)).unsqueeze(0).to(device)

#用预训练的vgg19模型 要网络下载权重
net = models.vgg19(pretrained=True).features.to(device).eval()

def show(a):
    a = a.cpu().detach().squeeze().numpy().transpose([1,2,0])
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    imshow(a)
    plt.show()

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=10000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = torch.optim.Adam([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    show(input_img)
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 200 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
                show(input_img)

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def greatershow(output):
    im = output.cpu().detach().squeeze().numpy().transpose([1,2,0])
    cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Image", im) 
    
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()


def generate_noise_image(content_image, noise_ratio = 0.6):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, 300, 400)).astype('float32')
    
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return torch.tensor(input_image).to(device)


#两种图片生成方式啊 注销掉的是老师的加噪音的方式
# t_img = copy.deepcopy(content_img.cpu())
# input_img = generate_noise_image(t_img.numpy())
# input_img=torch.randn(content_img.size()).to(device)
# cv2.namedWindow("Image") 
#这里直接拿原图搞了
#还有一种直接生成噪音图 torch.randn 
input_img = copy.deepcopy(style_img)
#没GPU慢的话改下steps，
output = run_style_transfer(net, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img,num_steps=13000,style_weight=1000000)
show(output.cpu().detach())
photo = output.cpu().detach().squeeze().numpy().transpose([1,2,0])
cv2.imwrite('save.jpg',photo*255)