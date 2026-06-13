import torch
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self, z_dim=100, features_g=64, img_channels=3):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),  
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.net(x)
class Critic(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                img_channels, features_d, kernel_size=4, stride=2, padding=1
            ), 
            nn.LeakyReLU(0.2, inplace=True),
            self._block(features_d, features_d * 2, 4, 2, 1), 
            self._block(features_d * 2, features_d * 4, 4, 2, 1), 
            self._block(features_d * 4, features_d * 8, 4, 2, 1), 
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0), 
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True), 
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.net(x)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
