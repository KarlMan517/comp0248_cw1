import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# Convolutional Block for Backbone
class ConvBlockPool(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Backbone
class Backbone(nn.Module):
    def __init__(self, base=32):
        super().__init__()

        # RGB
        self.rgb_l1 = ConvBlockPool(3, base)
        self.rgb_l2 = ConvBlockPool(base, base * 2)
        self.rgb_l3 = ConvBlockPool(base * 2, base * 4)
        self.rgb_l4 = ConvBlockPool(base * 4, base * 4, pool=False)

        # Depth (2ch)
        self.d_l1 = ConvBlockPool(2, base)
        self.d_l2 = ConvBlockPool(base, base * 2)
        self.d_l3 = ConvBlockPool(base * 2, base * 4)
        self.d_l4 = ConvBlockPool(base * 4, base * 4, pool=False)

        # Fusion
        self.fuse1 = nn.Conv2d(base * 2, base, 1)
        self.fuse2 = nn.Conv2d(base * 4, base * 2, 1)
        self.fuse3 = nn.Conv2d(base * 8, base * 4, 1)
        self.fuse4 = nn.Conv2d(base * 8, base * 4, 1)

        self.out_channel = base * 4  # f4 channels

    def forward(self, x):
        rgb = x[:, :3]
        depth = x[:, 3:5]

        f1_rgb = self.rgb_l1(rgb)
        f1_d = self.d_l1(depth)
        f1 = F.relu(self.fuse1(torch.cat([f1_rgb, f1_d], dim=1)))

        f2_rgb = self.rgb_l2(f1_rgb)
        f2_d = self.d_l2(f1_d)
        f2 = F.relu(self.fuse2(torch.cat([f2_rgb, f2_d], dim=1)))

        f3_rgb = self.rgb_l3(f2_rgb)
        f3_d = self.d_l3(f2_d)
        f3 = F.relu(self.fuse3(torch.cat([f3_rgb, f3_d], dim=1)))

        f4_rgb = self.rgb_l4(f3_rgb)
        f4_d = self.d_l4(f3_d)
        f4 = F.relu(self.fuse4(torch.cat([f4_rgb, f4_d], dim=1)))

        # depth residual
        f4 = f4 + 0.5 * f4_d

        feat_vec = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        return f1, f2, f3, f4, feat_vec




# Classification Head
class ImageClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes=10):
        super().__init__()
        C = in_dim
        self.head = nn.Sequential(
            nn.Linear(C, C * 2),
            nn.BatchNorm1d(C * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(C * 2, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, num_classes),
        )

    def forward(self, feat_vec):
        return self.head(feat_vec)




# SE Block
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# Convolutional Block for Segmentation Head
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# Segmentation Head
class SegmentationHead(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        C1 = base
        C2 = base * 2
        C3 = base * 4
        C4 = base * 4

        self.se1 = SEBlock(C1)
        self.se2 = SEBlock(C2)
        self.se3 = SEBlock(C3)
        self.se4 = SEBlock(C4)

        self.dec4 = ConvBlock(C4 + C3, C3)
        self.dec3 = ConvBlock(C3 + C2, C2)
        self.dec2 = ConvBlock(C2 + C1, C1)
        self.dec1 = ConvBlock(C1, C1 // 2)

        self.dropout = nn.Dropout2d(0.1)
        self.final_conv = nn.Conv2d(C1 // 2, 1, 1)

    def forward(self, f1, f2, f3, f4):
        f1 = self.se1(f1)
        f2 = self.se2(f2)
        f3 = self.se3(f3)
        f4 = self.se4(f4)

        z = torch.cat([f4, f3], dim=1)
        z = self.dec4(z)

        z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        z = torch.cat([z, f2], dim=1)
        z = self.dec3(z)

        z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        z = torch.cat([z, f1], dim=1)
        z = self.dec2(z)

        z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        z = self.dec1(z)

        z = self.dropout(z)
        seg_logits = self.final_conv(z)
        return seg_logits, z

# Detection Head
class BBoxHead(nn.Module):
    def __init__(self, C=128, seg_channels=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),

            nn.Conv2d(C, C // 2, 3, padding=1),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(C // 2, C // 4, 3, padding=1),
            nn.BatchNorm2d(C // 4),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(7)

        attn_dim = C // 4
        self.q_proj = nn.Conv2d(C // 4, attn_dim, 1)
        self.k_proj = nn.Conv2d(seg_channels, attn_dim, 1)
        self.v_proj = nn.Conv2d(seg_channels, attn_dim, 1)
        self.attn_out = nn.Conv2d(attn_dim, C // 4, 1)

        flat = (C // 4) * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(flat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, f4, seg_feat):
        bbox_feat = self.conv(f4)
        bbox_feat = self.pool(bbox_feat)  # (B, C//4, 7, 7)

        seg_resized = F.adaptive_avg_pool2d(seg_feat, (7, 7))

        Q = self.q_proj(bbox_feat)
        K = self.k_proj(seg_resized)
        V = self.v_proj(seg_resized)

        B, C_attn, H, W = Q.shape
        Q = Q.view(B, C_attn, -1)
        K = K.view(B, C_attn, -1)
        V = V.view(B, C_attn, -1)

        Q = F.normalize(Q, dim=1)
        K = F.normalize(K, dim=1)

        attn = torch.softmax(torch.bmm(Q.transpose(1, 2), K) / math.sqrt(C_attn), dim=-1)
        attn_out = torch.bmm(attn, V.transpose(1, 2))
        attn_out = attn_out.transpose(1, 2).view(B, C_attn, H, W)

        attn_out = self.attn_out(attn_out)
        bbox_feat = bbox_feat + 0.5 * attn_out

        out = self.fc(bbox_feat.flatten(1))

        cx, cy, w, h = torch.split(out, 1, dim=1)
        cx = torch.sigmoid(cx)
        cy = torch.sigmoid(cy)
        w = torch.sigmoid(w)
        h = torch.sigmoid(h)

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        bbox = torch.cat([x1, y1, x2, y2], dim=1)
        bbox = torch.clamp(bbox, 0.0, 1.0)
        return bbox



class MultiTaskModel(nn.Module):
    def __init__(self, base=32, num_classes=10):
        super().__init__()
        self.backbone = Backbone(base=base)
        self.cls_head = ImageClassifierHead(self.backbone.out_channel, num_classes=num_classes)
        self.seg_head = SegmentationHead(base=base)
        self.bbox_head = BBoxHead(C=self.backbone.out_channel, seg_channels=base // 2)

        # uncertainty weights (3 tasks)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        f1, f2, f3, f4, feat_vec = self.backbone(x)
        cls_logits = self.cls_head(feat_vec)
        seg_logits, seg_feat = self.seg_head(f1, f2, f3, f4)
        bbox = self.bbox_head(f4, seg_feat)
        return cls_logits, seg_logits, bbox