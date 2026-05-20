import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes."""

    def __init__(self, load_weights=False):
        super().__init__()
        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self._init_from_vgg(vgg)
        else:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _init_from_vgg(self, vgg):
        features = list(vgg.features.children())
        vgg_layers = [l for l in features if not isinstance(l, nn.MaxPool2d)]
        csr_layers = [l for l in self.frontend.children() if not isinstance(l, nn.MaxPool2d)]
        for v, c in zip(vgg_layers, csr_layers):
            if isinstance(v, nn.Conv2d) and isinstance(c, nn.Conv2d):
                c.weight.data = v.weight.data
                c.bias.data = v.bias.data

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_channels=3, dilation=False):
        layers = []
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                d = 2 if dilation else 1
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
                layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
