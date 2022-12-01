import torch
import torch.nn as nn

class CNN(nn.Module):
    
    def __init__(self, input_dimension, encoder_depth, const) -> None:
        super().__init__()
        self.model = self.construct_encoder(input_dimension, encoder_depth)
        if const is not None:
            self.const = nn.Parameter(torch.ones(1) * const)
        else:
            self.const = 1
    CHANNELS = [32, 64, 128, 256]


    def forward(self, x):
        y = torch.sigmoid(self.model(x))
        return y * self.const

    def construct_encoder(self, input_dimension, encoder_depth):
        channels = [input_dimension] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []
        for i in range(len(channels)- 1):
            blocks.append(nn.Conv2d(channels[i],channels[i+1],3,1,1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
        return nn.Sequential(*blocks[:-1])