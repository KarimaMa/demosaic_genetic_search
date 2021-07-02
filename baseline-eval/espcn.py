import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, 1 * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid()
        )

    def _initialize(self):
        for p in self.feature_maps:
            if hasattr(p, "weight"):
                nn.init.xavier_normal_(p.weight)
                # nn.init.normal_(p.weight, mean=0.0, std=0.001)
                # nn.init.zeros_(p.bias)

        for p in self.sub_pixel:
            if hasattr(p, "weight"):
                nn.init.xavier_normal_(p.weight)
                # nn.init.normal_(p.weight, mean=0.0, std=0.001)
                # nn.init.zeros_(p.bias)

    def forward(self, inputs):
        out = self.feature_maps(inputs)
        out = self.sub_pixel(out)
        return out
