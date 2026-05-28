"""Stage 2 lead model with cross-lead fusion.

This module is the custom Stage 2 model used for the ECG image digitization
project. It processes four rectified ECG lead-row crops with a shared U-Net
encoder/decoder and optionally fuses features across the four rows.

Supported fusion modes:
- none: shared encoder/decoder only.
- conv2d: per-lead channel reduction, concatenation, and 2D mixing.
- shared_conv2d: parameter-efficient shared reduction before 2D mixing.
- cross_attn: multi-head self-attention across the four lead rows at deep levels.

Final attention experiments used fusion_type='cross_attn' and fusion_levels=[3, 4].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "segmentation_models_pytorch is required. Install with: "
        "pip install segmentation-models-pytorch"
    ) from exc


class CrossLeadConvFusion(nn.Module):
    """Conv2D feature fusion across four ECG lead rows."""

    def __init__(self, channels: int, num_leads: int = 4, shared: bool = False, reduction_ratio: int = 4):
        super().__init__()
        self.channels = channels
        self.num_leads = num_leads
        self.shared = shared
        reduced = max(1, channels // reduction_ratio)
        self.reduced_channels = reduced

        def reduce_block() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced),
                nn.ReLU(inplace=True),
            )

        if shared:
            self.reduce_conv = reduce_block()
        else:
            self.reduce_convs = nn.ModuleList([reduce_block() for _ in range(num_leads)])

        concat_channels = reduced * num_leads
        self.mix_conv = nn.Sequential(
            nn.Conv2d(concat_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        batch = batch_size
        _, channels, height, width = x.shape
        x_leads = x.view(batch, self.num_leads, channels, height, width)

        if self.shared:
            reduced = [self.reduce_conv(x_leads[:, i]) for i in range(self.num_leads)]
        else:
            reduced = [self.reduce_convs[i](x_leads[:, i]) for i in range(self.num_leads)]

        mixed = self.mix_conv(torch.cat(reduced, dim=1))
        mixed = mixed.unsqueeze(1).expand(batch, self.num_leads, channels, height, width)
        return (x_leads + mixed).reshape(batch * self.num_leads, channels, height, width)


class CrossLeadAttentionFusion(nn.Module):
    """Cross-lead self-attention at each spatial feature location.

    Input shape is (B*L, C, H, W). The tensor is reshaped into
    (B*H*W, L, C), so the four lead rows are the attention tokens for each
    spatial location. This keeps the sequence length small while letting the
    model exchange information across correlated ECG rows.
    """

    def __init__(self, channels: int, num_leads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_leads = num_leads
        self.num_heads = next((h for h in (8, 4, 2, 1) if channels % h == 0), 1)

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor, batch_size: int | None = None, batchsize: int | None = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = batchsize
        if batch_size is None:
            raise ValueError("batch_size must be provided")

        batch = batch_size
        _, channels, height, width = x.shape
        x_leads = x.view(batch, self.num_leads, channels, height, width)
        x_hw = x_leads.permute(0, 3, 4, 1, 2)          # (B, H, W, L, C)
        x_seq = x_hw.reshape(batch * height * width, self.num_leads, channels)

        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.ffn(x_seq)

        x_hw = x_seq.reshape(batch, height, width, self.num_leads, channels)
        x_out = x_hw.permute(0, 3, 4, 1, 2)
        return x_out.reshape(batch * self.num_leads, channels, height, width)


class Net(nn.Module):
    """Shared Stage 2 lead model.

    Args:
        encoder_name: SMP encoder name, e.g. 'resnet34' or 'tu-efficientnet_b6'.
        encoder_weights: optional pretrained encoder weights.
        fusion_type: 'none', 'conv2d', 'shared_conv2d', or 'cross_attn'.
        fusion_levels: encoder levels to fuse. Attention should use [3, 4].
        num_leads: number of lead-row crops. The project uses four rows.
        loss_weight: positive class weight for BCE.

    Expected batch format:
        batch['image']: uint8 tensor with shape (B, 4, 3, H, W)
        batch['pixel']: float tensor with shape (B, 4, 1, H, W), training only
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        fusion_type: str | None = "conv2d",
        fusion_levels: list[int] | tuple[int, ...] = (1, 2, 3, 4),
        num_leads: int = 4,
        loss_weight: float = 10.0,
        **_: object,
    ):
        super().__init__()
        fusion_type = "none" if fusion_type is None else fusion_type.lower()
        valid = {"none", "conv2d", "shared_conv2d", "cross_attn"}
        if fusion_type not in valid:
            raise ValueError(f"fusion_type must be one of {sorted(valid)}, got {fusion_type!r}")

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.fusion_type = fusion_type
        self.fusion_levels = list(fusion_levels)
        self.num_leads = num_leads
        self.loss_weight = loss_weight
        self.output_type = ["infer", "loss"]

        self.register_buffer("D", torch.tensor(0))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            decoder_channels=[256, 128, 64, 32, 16],
        )
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.pixel_head = nn.Conv2d(16, 1, kernel_size=1)
        self.dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

        self.fusion_modules = nn.ModuleDict()
        if self.fusion_type != "none":
            encoder_channels = self.encoder.out_channels
            for level in self.fusion_levels:
                channels = encoder_channels[level + 1]
                if self.fusion_type == "cross_attn":
                    if level < 3:
                        raise ValueError("cross_attn is memory-heavy; use fusion_levels=[3, 4].")
                    module = CrossLeadAttentionFusion(channels=channels, num_leads=num_leads)
                else:
                    module = CrossLeadConvFusion(
                        channels=channels,
                        num_leads=num_leads,
                        shared=(self.fusion_type == "shared_conv2d"),
                    )
                self.fusion_modules[f"fusion_{level}"] = module

    def forward(self, batch: dict[str, torch.Tensor], L: int | None = None) -> dict[str, torch.Tensor]:
        del L
        device = self.D.device
        image = batch["image"].to(device)
        batch_size, num_leads, channels, height, width = image.shape

        x = image.float() / 255.0
        x = (x - self.mean) / self.std
        x = x.view(batch_size * num_leads, channels, height, width)

        features = list(self.encoder(x))
        if self.fusion_type != "none":
            for level in self.fusion_levels:
                key = f"fusion_{level}"
                if key in self.fusion_modules:
                    features[level + 1] = self.fusion_modules[key](features[level + 1], batch_size=batch_size)

        decoded = self.decoder(features)
        pixel = self.pixel_head(decoded).view(batch_size, num_leads, 1, height, width)

        output: dict[str, torch.Tensor] = {}
        needs_target = "loss" in self.output_type or "dice_loss" in self.output_type
        if needs_target:
            target = batch["pixel"].to(device)
            pixel_flat = pixel.view(batch_size * num_leads, 1, height, width)
            target_flat = target.view(batch_size * num_leads, 1, height, width)

        if "loss" in self.output_type:
            pos_weight = torch.tensor([self.loss_weight], device=device)
            output["pixel_loss"] = F.binary_cross_entropy_with_logits(pixel_flat, target_flat, pos_weight=pos_weight)

        if "dice_loss" in self.output_type:
            output["pixel_dice_loss"] = self.dice_loss_fn(pixel_flat, target_flat)

        if "infer" in self.output_type:
            output["pixel"] = torch.sigmoid(pixel)

        return output


def run_smoke_test(device: str = "cuda") -> None:
    """Minimal shape test for the current Net API."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    batch_size, num_leads, height, width = 1, 4, 128, 256
    batch = {
        "image": torch.randint(0, 256, (batch_size, num_leads, 3, height, width), dtype=torch.uint8),
        "pixel": torch.rand(batch_size, num_leads, 1, height, width),
    }
    net = Net(encoder_name="resnet34", encoder_weights=None, fusion_type="cross_attn", fusion_levels=[3, 4]).to(device)
    net.output_type = ["loss", "dice_loss", "infer"]
    with torch.no_grad():
        output = net(batch)
    assert output["pixel"].shape == (batch_size, num_leads, 1, height, width)
    print("Smoke test passed:", output["pixel"].shape)


if __name__ == "__main__":
    run_smoke_test(device="cuda" if torch.cuda.is_available() else "cpu")
