"""
services/vision/us_thyroid/arch.py
Model architecture for Thyroid Ultrasound (TN3K dataset).

Identical to us_breast/arch.py, only changes:
  - NUM_CLASSES = 2  (benign=0, malignant=1)
  - MEAN/STD     = values computed from the TN3K training set
  - IDX_TO_CLASS / CLASS_TO_IDX updated for thyroid

Not copied:
  - CapsuleLayer, CapsuleNetwork   (CLASSIFICATION_HEAD='fc', unused)
  - DeformableConvBlock            (USE_Deform=False, unused)
  - Dataset/transform classes      (training only - see the training notebook)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class Config:
    """
    Centralized config for us_thyroid inference.

    Mean/Std computed from TN3K training fold 0.
    """
    # Architecture
    MODEL_TYPE            = 'multitask'
    BACKBONE              = 'efficientnet_b4'
    NUM_CLASSES           = 2            # TN3K: 0=benign, 1=malignant
    USE_Deform            = False
    CLASSIFICATION_HEAD   = 'fc'

    # Inference
    IMG_SIZE = 256
    DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TN3K dataset normalization - computed from trainval fold 0
    # Update with real values after running the notebook
    MEAN = [0.2830, 0.2830, 0.2830]
    STD  = [0.1950, 0.1950, 0.1950]

    # Class mapping - TN3K: binary classification
    IDX_TO_CLASS = {0: 'benign', 1: 'malignant'}
    CLASS_TO_IDX = {'benign': 0, 'malignant': 1}

    # Training parameters (not used at inference time)
    BATCH_SIZE      = 16
    EPOCHS          = 50
    LEARNING_RATE   = 1e-4
    SEED            = 42



class ConvBlock(nn.Module):
    """
    Double conv block for the UNet decoder path.
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)



class UNet_MTL(nn.Module):
    """
    Multi-task UNet with an EfficientNet-B4 encoder.

    Forward pass returns (seg_output, cls_output, bottleneck_out):
      - seg_output:     (B, 1, H, W)       - sigmoid mask [0, 1]
      - cls_output:     (B, NUM_CLASSES)   - raw logits (before softmax)
      - bottleneck_out: (B, 448, h, w)     - bottleneck feature map

    EfficientNet-B4 encoder channels: [24, 32, 56, 160, 448]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.BACKBONE, pretrained=False, features_only=True
        )
        ch = self.backbone.feature_info.channels()  # [24,32,56,160,448]

        self.upconv4 = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.dec4    = ConvBlock(ch[3] + ch[3], ch[3])

        self.upconv3 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3    = ConvBlock(ch[2] + ch[2], ch[2])

        self.upconv2 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2    = ConvBlock(ch[1] + ch[1], ch[1])

        self.upconv1 = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec1    = ConvBlock(ch[0] + ch[0], ch[0])

        # Segmentation head and classification head
        self.seg_head = nn.Conv2d(ch[0], 1, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[4], 512),
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
            bottleneck_out: (B, 448, h, w)       - encoder bottleneck
        """
        e1, e2, e3, e4, bot = self.backbone(x)

        # Classification from the bottleneck
        cls_output = self.cls_head(bot)

        # Decoder (segmentation branch)
        d4 = self.dec4(torch.cat([self.upconv4(bot), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2),  e1], dim=1))

        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        seg_output = torch.sigmoid(self.seg_head(d1))

        return seg_output, cls_output, bot
