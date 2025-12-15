import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """
    A small ResNet BasicBlock for 1D signals.
    Keeps temporal length when stride=1; supports dilation for larger receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        norm_groups: int = 8,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length with symmetric padding")
        padding = (kernel_size // 2) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups=min(norm_groups, out_channels), num_channels=out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=min(norm_groups, out_channels), num_channels=out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.GroupNorm(num_groups=min(norm_groups, out_channels), num_channels=out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act(out)
        return out


class ResNet1DBackbone(nn.Module):
    """
    Simple 1D ResNet backbone for (B,2,T) IQ input.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 256),
        stage_dilations: tuple[int, int, int, int] = (1, 2, 4, 8),
        norm_groups: int = 8,
    ):
        super().__init__()
        assert len(layers) == 4 and len(stage_channels) == 4 and len(stage_dilations) == 4

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(num_groups=min(norm_groups, base_channels), num_channels=base_channels),
            nn.ReLU(inplace=True),
        )

        ch = base_channels
        self.stage1, ch = self._make_stage(ch, stage_channels[0], layers[0], stage_dilations[0], norm_groups)
        self.stage2, ch = self._make_stage(ch, stage_channels[1], layers[1], stage_dilations[1], norm_groups)
        self.stage3, ch = self._make_stage(ch, stage_channels[2], layers[2], stage_dilations[2], norm_groups)
        self.stage4, ch = self._make_stage(ch, stage_channels[3], layers[3], stage_dilations[3], norm_groups)
        self.out_channels = ch

    @staticmethod
    def _make_stage(
        in_ch: int, out_ch: int, n_blocks: int, dilation: int, norm_groups: int
    ) -> tuple[nn.Sequential, int]:
        blocks = [BasicBlock1D(in_ch, out_ch, dilation=dilation, norm_groups=norm_groups)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock1D(out_ch, out_ch, dilation=dilation, norm_groups=norm_groups))
        return nn.Sequential(*blocks), out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class SignalSeparator(nn.Module):
    """
    Drop-in replacement for `model_complex.SignalSeparator`.

    Input:
      x: (B, 2, T) float32
    Output:
      list of 4 tensors, each (B, 1, T):
        [sig1_real, sig1_imag, sig2_real, sig2_imag]
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 256),
        stage_dilations: tuple[int, int, int, int] = (1, 2, 4, 8),
        norm_groups: int = 8,
    ):
        super().__init__()
        self.backbone = ResNet1DBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=layers,
            stage_channels=stage_channels,
            stage_dilations=stage_dilations,
            norm_groups=norm_groups,
        )
        self.head = nn.Conv1d(self.backbone.out_channels, 4, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)  # (B,C,T)
        y = self.head(feat)      # (B,4,T)
        return list(torch.chunk(y, 4, dim=1))


