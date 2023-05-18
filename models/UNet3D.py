import torch
from torch import nn


class ConvX2(nn.Module):
    def __init__(self, input_channels, corresponding_channels, group_norm_size=8, pre_norm=True):
        super(ConvX2, self).__init__()
        if input_channels < group_norm_size:
            first_input_norm = 1
        else:
            first_input_norm = group_norm_size

        if pre_norm:
            self.conv_x2 = nn.Sequential(
                nn.GroupNorm(first_input_norm, input_channels),
                # nn.BatchNorm3d(input_channels),
                nn.Conv3d(input_channels, corresponding_channels, kernel_size=(3, 3, 3), padding='same'),
                # nn.ReLU(inplace=True),
                nn.PReLU(),
                # will do the activation in-place without creating separate space for computation
                nn.GroupNorm(group_norm_size, corresponding_channels),
                # nn.BatchNorm3d(corresponding_channels),
                nn.Conv3d(corresponding_channels, corresponding_channels, kernel_size=(3, 3, 3), padding='same'),
                # nn.ReLU(inplace=True),
                nn.PReLU()
                # will do the activation in-place without creating separate space for computation
            )
        else:
            self.conv_x2 = nn.Sequential(
                nn.Conv3d(input_channels, corresponding_channels, kernel_size=(3, 3, 3), padding='same'),
                nn.ReLU(inplace=True),
                # will do the activation in-place without creating separate space for computation
                nn.GroupNorm(group_norm_size, corresponding_channels),
                nn.Conv3d(corresponding_channels, corresponding_channels, kernel_size=(3, 3, 3), padding='same'),
                nn.ReLU(inplace=True),
                # will do the activation in-place without creating separate space for computation
                nn.GroupNorm(group_norm_size, corresponding_channels),  # BatchNorm2D can be added here
            )

    def forward(self, x):
        x = self.conv_x2(x)
        return x


class DownConv(nn.Module):
    def __init__(self, input_channels, corresponding_channels):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            ConvX2(input_channels, corresponding_channels)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose3d(input_channels, input_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_x2 = ConvX2(input_channels, output_channels)

    def forward(self, x, step_x):
        x = self.up_conv(x, output_size=step_x.size())
        x = torch.cat([step_x, x], dim=1)  # combine the two inputs on column dimension
        return self.conv_x2(x)


class OutputConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OutputConv, self).__init__()
        self.conv_1x = nn.Conv3d(input_channels, output_channels, kernel_size=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.conv_1x(x)
        # x = self.softmax(x)
        # x = self.sigmoid(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet3D, self).__init__()

        # Defining the layers and sequence of execution
        self.input_step = ConvX2(input_channels, 64)
        # doubling the features everytime a downward step is taken
        self.down_step1 = DownConv(64, 128)
        self.down_step2 = DownConv(128, 256)
        self.down_step3 = DownConv(256, 512)
        self.down_step4 = DownConv(512, 1024)
        # upward step halves the features in the previous layer
        self.up_step1 = UpConv(1024, 512)
        self.up_step2 = UpConv(512, 256)
        self.up_step3 = UpConv(256, 128)
        self.up_step4 = UpConv(128, 64)
        self.output_step = OutputConv(64, output_channels)

        self.initialize_params()

    def forward(self, x):
        # decoder steps
        x_layer1 = self.input_step(x)
        x_layer2 = self.down_step1(x_layer1)
        x_layer3 = self.down_step2(x_layer2)
        x_layer4 = self.down_step3(x_layer3)
        x_layer5 = self.down_step4(x_layer4)
        x = self.up_step1(x_layer5, x_layer4)
        x = self.up_step2(x, x_layer3)
        x = self.up_step3(x, x_layer2)
        x = self.up_step4(x, x_layer1)
        x = self.output_step(x)

        return x

    def initialize_params(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                kernel_size = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
                N = kernel_size * module.in_channels
                std = float(torch.sqrt(torch.tensor(2/N)))
                module.weight.data = torch.normal(mean=0, std=std, size=module.weight.shape, device='cuda')


