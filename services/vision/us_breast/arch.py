"""
services/vision/us_breast/arch.py
Model architecture classes - copied from busi-architecture-full-implementation.ipynb.

Only keeps what is needed for the POC (checkpoint mtl_effnet_fc_conv.pt):
  - Config
  - ConvBlock
  - UNet_MTL      <- main model, used for inference
  - UNet_Segmentation  <- kept for reference, not used in the POC

NOT copied:
  - CapsuleLayer, CapsuleNetwork  (cfg.CLASSIFICATION_HEAD='capsnet' - unused)
  - DeformableConvBlock           (cfg.USE_Deform=False - unused)
  - Dataset/transform classes     (training only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class Config:
    """
    Centralized config. Inference only needs the fields below.
    Training fields (EPOCHS, BATCH_SIZE, ...) are kept for reference.
    """
    # Architecture
    MODEL_TYPE = 'multitask'
    BACKBONE = 'efficientnet_b4'
    NUM_CLASSES = 3
    USE_Deform = False
    CLASSIFICATION_HEAD = 'fc'   # 'fc' | 'capsnet' - POC uses 'fc'

    # Inference
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training parameters (not used at inference time)
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    TEST_RATIO = 0.2
    SEED = 42

    # Normalization values computed from the BUSI training set (differs from ImageNet)
    MEAN = [0.2720, 0.2720, 0.2720]   # grayscale-like: 3 channels nearly equal
    STD  = [0.1890, 0.1890, 0.1890]

    # Class mapping following BUSI's alphabetical order
    IDX_TO_CLASS = {0: "benign", 1: "malignant", 2: "normal"}
    CLASS_TO_IDX = {"benign": 0, "malignant": 1, "normal": 2}



class ConvBlock(nn.Module):
    """
    Double conv block used in the UNet decoder path.
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# UNet_MTL - Multi-Task Learning: Segmentation + Classification

class UNet_MTL(nn.Module):
    """
    Multi-task UNet with an EfficientNet-B4 encoder.

    Forward pass returns (seg_output, cls_output, bottleneck_out):
      - seg_output:    (B, 1, H, W)   - sigmoid mask [0, 1]
      - cls_output:    (B, NUM_CLASSES) - raw logits (before softmax)
      - bottleneck_out: (B, 448, 7, 7) - feature map from the bottom of the
                        encoder, used to extract bottleneck_enriched features for the visual interpreter

    EfficientNet-B4 encoder channels: [24, 32, 56, 160, 448]
    """

    def __init__(self, cfg: Config):
        super(UNet_MTL, self).__init__()
        self.cfg = cfg

        # EfficientNet-B4 encoder, returns 5 feature scales
        self.backbone = timm.create_model(
            cfg.BACKBONE, pretrained=False, features_only=True
        )
        backbone_channels = self.backbone.feature_info.channels()

        # Decoder path
        conv_block = ConvBlock

        self.upconv4 = nn.ConvTranspose2d(backbone_channels[4], backbone_channels[3], kernel_size=2, stride=2)
        self.dec4    = conv_block(backbone_channels[3] + backbone_channels[3], backbone_channels[3])

        self.upconv3 = nn.ConvTranspose2d(backbone_channels[3], backbone_channels[2], kernel_size=2, stride=2)
        self.dec3    = conv_block(backbone_channels[2] + backbone_channels[2], backbone_channels[2])

        self.upconv2 = nn.ConvTranspose2d(backbone_channels[2], backbone_channels[1], kernel_size=2, stride=2)
        self.dec2    = conv_block(backbone_channels[1] + backbone_channels[1], backbone_channels[1])

        self.upconv1 = nn.ConvTranspose2d(backbone_channels[1], backbone_channels[0], kernel_size=2, stride=2)
        self.dec1    = conv_block(backbone_channels[0] + backbone_channels[0], backbone_channels[0])

        # Segmentation head and classification head (FC)
        self.seg_head = nn.Conv2d(backbone_channels[0], 1, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_channels[4], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) - normalized input image

        Returns:
            seg_output:     (B, 1, H, W)        - probability mask
            cls_output:     (B, NUM_CLASSES)     - raw logits
            bottleneck_out: (B, 448, 7, 7)       - encoder bottleneck
        """
        features = self.backbone(x)
        enc1_out, enc2_out, enc3_out, enc4_out, bottleneck_out = features

        # Classification from the bottleneck
        cls_output = self.cls_head(bottleneck_out)

        d4 = self.upconv4(bottleneck_out)
        d4 = torch.cat([d4, enc4_out], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, enc3_out], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, enc2_out], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, enc1_out], dim=1)
        d1 = self.dec1(d1)

        # Upsample the mask back to the original image size
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        seg_output = torch.sigmoid(self.seg_head(d1))

        return seg_output, cls_output, bottleneck_out


# UNet_Segmentation - Segmentation only (reference, not used in the POC)

class UNet_Segmentation(nn.Module):
    """Segmentation-only UNet. Kept for reference, not used for inference."""

    def __init__(self, cfg: Config):
        super(UNet_Segmentation, self).__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.BACKBONE, pretrained=False, features_only=True
        )
        backbone_channels = self.backbone.feature_info.channels()
        conv_block = ConvBlock

        self.upconv4 = nn.ConvTranspose2d(backbone_channels[4], backbone_channels[3], kernel_size=2, stride=2)
        self.dec4    = conv_block(backbone_channels[3] + backbone_channels[3], backbone_channels[3])

        self.upconv3 = nn.ConvTranspose2d(backbone_channels[3], backbone_channels[2], kernel_size=2, stride=2)
        self.dec3    = conv_block(backbone_channels[2] + backbone_channels[2], backbone_channels[2])

        self.upconv2 = nn.ConvTranspose2d(backbone_channels[2], backbone_channels[1], kernel_size=2, stride=2)
        self.dec2    = conv_block(backbone_channels[1] + backbone_channels[1], backbone_channels[1])

        self.upconv1 = nn.ConvTranspose2d(backbone_channels[1], backbone_channels[0], kernel_size=2, stride=2)
        self.dec1    = conv_block(backbone_channels[0] + backbone_channels[0], backbone_channels[0])

        self.seg_head = nn.Conv2d(backbone_channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        enc1_out, enc2_out, enc3_out, enc4_out, bottleneck_out = features

        d4 = self.upconv4(bottleneck_out)
        d4 = torch.cat([d4, enc4_out], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, enc3_out], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, enc2_out], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, enc1_out], dim=1)
        d1 = self.dec1(d1)

        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(self.seg_head(d1))
