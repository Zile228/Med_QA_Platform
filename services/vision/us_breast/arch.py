"""
services/vision/us_breast/arch.py
==================================
Model architecture classes - copied từ busi-architecture-full-implementation.ipynb.

Chỉ giữ lại những gì cần cho POC (checkpoint mtl_effnet_fc_conv.pt):
  - Config
  - ConvBlock
  - UNet_MTL      <- model chính, dùng cho inference
  - UNet_Segmentation  <- giữ lại để tham chiếu, không dùng trong POC

KHÔNG copy:
  - CapsuleLayer, CapsuleNetwork  (cfg.CLASSIFICATION_HEAD='capsnet' - không dùng)
  - DeformableConvBlock           (cfg.USE_Deform=False - không dùng)
  - Dataset/transform classes     (training only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class Config:
    """
    Centralized config. Inference chỉ cần các field bên dưới.
    Training fields (EPOCHS, BATCH_SIZE, ...) giữ lại cho reference.
    """
    # Architecture
    MODEL_TYPE = 'multitask'
    BACKBONE = 'efficientnet_b4'
    NUM_CLASSES = 3
    USE_Deform = False
    CLASSIFICATION_HEAD = 'fc'   # 'fc' | 'capsnet' - POC dùng 'fc'

    # Inference
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Tham so training (khong dung luc inference)
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    TEST_RATIO = 0.2
    SEED = 42

    # Gia tri chuan hoa tinh tu BUSI training set (khac ImageNet)
    MEAN = [0.2720, 0.2720, 0.2720]   # grayscale-like: 3 channels gần bằng nhau
    STD  = [0.1890, 0.1890, 0.1890]

    # Anh xa class theo thu tu alphabet cua BUSI
    IDX_TO_CLASS = {0: "benign", 1: "malignant", 2: "normal"}
    CLASS_TO_IDX = {"benign": 0, "malignant": 1, "normal": 2}



class ConvBlock(nn.Module):
    """
    Double conv block dùng trong decoder path của UNet.
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
    Multi-task UNet với EfficientNet-B4 encoder.

    Forward pass trả về (seg_output, cls_output, bottleneck_out):
      - seg_output:    (B, 1, H, W)   - sigmoid mask [0, 1]
      - cls_output:    (B, NUM_CLASSES) - raw logits (trước softmax)
      - bottleneck_out: (B, 448, 7, 7) - feature map từ đáy encoder
                        dùng để extract bottleneck_features cho LLM

    EfficientNet-B4 encoder channels: [24, 32, 56, 160, 448]
    """

    def __init__(self, cfg: Config):
        super(UNet_MTL, self).__init__()
        self.cfg = cfg

        # EfficientNet-B4 encoder, tra ve 5 feature scale
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

        # Segmentation head va classification head (FC)
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

        # Classification tu bottleneck
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

        # Upsample mask ve kich thuoc anh goc
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        seg_output = torch.sigmoid(self.seg_head(d1))

        return seg_output, cls_output, bottleneck_out


# UNet_Segmentation - Segmentation only (reference, không dùng POC)

class UNet_Segmentation(nn.Module):
    """Segmentation-only UNet. Giu lai de tham chieu, khong dung cho inference."""

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
