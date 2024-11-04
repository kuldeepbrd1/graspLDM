import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Attention(nn.Module):
    # Adapted from https://github.com/alexzhou907/PVD
    # Used for global attention over context vectors like pc shape latent
    def __init__(self, in_ch, num_groups, D=3):
        super(Attention, self).__init__()
        assert in_ch % num_groups == 0
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)
        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)

        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)

    def forward(self, x):
        B, C = x.shape[:2]
        h = x

        q = self.q(h).reshape(B, C, -1)
        k = self.k(h).reshape(B, C, -1)
        v = self.v(h).reshape(B, C, -1)

        qk = torch.matmul(q.permute(0, 2, 1), k)  # * (int(C) ** (-0.5))

        w = self.sm(qk)

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B, C, *x.shape[2:])

        h = self.out(h)

        x = h + x

        x = self.nonlin(self.norm(x))

        return x


class FCLayers(nn.Module):
    def __init__(
        self,
        in_features,
        layer_outs_specs=[128, 256, 512],
        layer_normalization=True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = layer_outs_specs[-1]

        self.layer_specs = layer_outs_specs
        self.layer_normalization = layer_normalization

        self.num_layers = len(layer_outs_specs)
        self.layers = self._build_layers()

    def _build_layers(self):
        module_list = []

        for idx, layer_out_features in enumerate(self.layer_specs):
            in_feats = self.in_features if idx == 0 else self.layer_specs[idx - 1]
            out_feats = layer_out_features

            if self.layer_normalization:
                module_list.append(
                    nn.Sequential(
                        nn.Linear(
                            in_feats, out_feats, bias=not self.layer_normalization
                        ),
                        nn.LayerNorm(out_feats),
                        nn.ReLU(),
                    )
                )
            else:
                module_list.append(
                    nn.Sequential(
                        nn.Linear(
                            in_feats, out_feats, bias=not self.layer_normalization
                        ),
                        nn.ReLU(),
                    )
                )

        return nn.Sequential(*module_list)

    def forward(self, x):
        return self.layers(x)
