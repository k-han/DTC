import torch.nn as nn

class VGG(nn.Module):
    cfg = {
        '4+2': [64, 'M', 128, 'M', 256, 'M', 256, 'M'],
        '5+1': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    }
    # these two 6-layer variants of VGG are following https://arxiv.org/abs/1711.10125 and https://github.com/GT-RIPL/L2C/blob/master/models/vgg.py
    def __init__(self, n_layer, out_dim=10, in_channels=3, img_sz=32):
        super(VGG, self).__init__()
        self.conv_func = nn.Conv2d
        self.features = self._make_layers(VGG.cfg[n_layer],in_channels)
        if n_layer=='4+2':
            self.feat_map_sz = img_sz // 16
            feat_dim = 256*(self.feat_map_sz**2)
            self.last = nn.Sequential(
                nn.Linear(feat_dim, feat_dim//2),
                nn.BatchNorm1d(feat_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim//2, out_dim)
            )
            self.last.in_features = feat_dim
        elif n_layer=='5+1':
            self.feat_map_sz = img_sz // 32
            self.last = nn.Linear(512*(self.feat_map_sz**2), out_dim)

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.last(x)
        return x, out 

