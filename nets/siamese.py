import torch
import torch.nn as nn

from nets.vgg import vgg16
from nets.alexnet import AlexNet


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2, 1]
        padding = [0, 0, 0, 0, 0, 0]
        stride = 2
        for i in range(6):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width) * get_output_length(height)


class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = vgg16(pretrained, 3)
        self.alex = AlexNet(3, init_weights=True)

        del self.vgg.avgpool
        del self.vgg.classifier
        del self.alex.avgpool
        del self.alex.classifier

        flat_shape = 768 * get_img_output_length(input_shape[1], input_shape[0])
        # print(flat_shape)
        # flat_shape = 768 * 4
        self.fully_connect1 = torch.nn.Linear(flat_shape, 768)
        self.fully_connect2 = torch.nn.Linear(768, 1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1_vgg = self.vgg.features(x1)
        # print(x1_vgg.shape)
        x1_vgg = self.max_pool(x1_vgg)
        # print(x1_vgg.shape)
        x1_alex = self.alex.features(x1)
        # print(x1_alex.shape)
        x1 = torch.cat((x1_vgg, x1_alex), 1)
        # print(x1.shape)

        x2_vgg = self.vgg.features(x2)
        x2_vgg = self.max_pool(x2_vgg)
        x2_alex = self.alex.features(x2)
        x2 = torch.cat((x2_vgg, x2_alex), 1)

        # -------------------------#
        #   相减取绝对值，取l1距离
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
