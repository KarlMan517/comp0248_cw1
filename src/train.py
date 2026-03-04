import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import FrameDataset, build_clip_index, build_frame_index, split_dataset
from model import MultiTaskModel
from utils import seed_everything


import torch
import torch.nn.functional as F



# Dice Loss
def dice_loss_with_logits(logits, targets, eps=1e-6):

    probs = torch.sigmoid(logits)

    num = 2.0 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps

    dice = 1.0 - (num + eps) / den

    return dice.mean()


# Segmentation Loss
def masked_seg_loss(seg_logits, mask, has_mask):

    valid = has_mask > 0.5

    if valid.sum() == 0:
        return seg_logits.new_tensor(0.0)

    seg_logits_v = seg_logits[valid]
    mask_v = mask[valid]

    bce = F.binary_cross_entropy_with_logits(seg_logits_v, mask_v)

    dsc = dice_loss_with_logits(seg_logits_v, mask_v)

    return bce + dsc



# Bounding Box Loss
def masked_bbox_loss_with_iou(bbox_pred, bbox_gt, has_bbox):

    valid = has_bbox > 0.5

    if valid.sum() == 0:
        return bbox_pred.new_tensor(0.0)

    pred = bbox_pred[valid]
    gt = bbox_gt[valid]

    # Smooth L1 regression
    l1 = F.smooth_l1_loss(pred, gt)

    # IoU
    x1 = torch.max(pred[:,0], gt[:,0])
    y1 = torch.max(pred[:,1], gt[:,1])
    x2 = torch.min(pred[:,2], gt[:,2])
    y2 = torch.min(pred[:,3], gt[:,3])

    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    area_p = (pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])
    area_g = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])

    union = area_p + area_g - inter + 1e-6

    iou = inter / union

    iou_loss = 1.0 - iou.mean()

    return l1 + iou_loss

# Segmentation IoU
def compute_seg_iou(seg_logits, mask, has_mask):

    probs = torch.sigmoid(seg_logits)
    pred = (probs > 0.5).float()

    inter = (pred * mask).sum(dim=(1,2,3))
    union = (pred + mask - pred*mask).sum(dim=(1,2,3)) + 1e-6

    iou = inter / union
    return iou[has_mask > 0.5]

# Bounding Box IoU
def compute_bbox_iou(bbox_pred, bbox_gt, has_bbox):

    valid = has_bbox > 0.5

    if valid.sum() == 0:
        return torch.tensor([], device=bbox_pred.device)

    pred = bbox_pred[valid]
    gt = bbox_gt[valid]

    x1 = torch.max(pred[:,0], gt[:,0])
    y1 = torch.max(pred[:,1], gt[:,1])
    x2 = torch.min(pred[:,2], gt[:,2])
    y2 = torch.min(pred[:,3], gt[:,3])

    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    area_p = (pred[:,2]-pred[:,0])*(pred[:,3]-pred[:,1])
    area_g = (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])

    union = area_p + area_g - inter + 1e-6

    return inter / union


# Validation
@torch.no_grad()
def validate(model, loader, device, criterion_cls, stage):

    model.eval()

    total_loss = 0
    correct = 0
    total_samples = 0

    seg_ious = []
    bbox_ious = []

    val_bar = tqdm(loader, desc="Validation", leave=False)

    for batch in val_bar:

        x = batch["image"].to(device)
        y = batch["label"].to(device)

        mask = batch["mask"].to(device)
        bbox_gt = batch["bbox"].to(device)

        has_mask = batch["has_mask"].to(device)
        has_bbox = batch["has_bbox"].to(device)

        cls_logits, seg_logits, bbox_pred = model(x)

        L_cls = criterion_cls(cls_logits, y)
        L_seg = masked_seg_loss(seg_logits, mask, has_mask)

        if stage == 1:

            total = L_cls + L_seg

        else:

            L_bbox = masked_bbox_loss_with_iou(bbox_pred, bbox_gt, has_bbox)

            total = L_cls + L_seg + 0.5 * L_bbox

            bbox_iou = compute_bbox_iou(bbox_pred, bbox_gt, has_bbox)

            if len(bbox_iou) > 0:
                bbox_ious.extend(bbox_iou.cpu().tolist())

        total_loss += float(total)

        preds = cls_logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total_samples += y.size(0)

        seg_iou = compute_seg_iou(seg_logits, mask, has_mask)

        if len(seg_iou) > 0:
            seg_ious.extend(seg_iou.cpu().tolist())

    total_loss /= len(loader)

    val_acc = correct / total_samples

    mean_seg_iou = np.mean(seg_ious) if len(seg_ious) > 0 else 0
    mean_bbox_iou = np.mean(bbox_ious) if len(bbox_ious) > 0 else 0

    return total_loss, val_acc, mean_seg_iou, mean_bbox_iou



# Training
def train_two_stage(
        train_index,
        val_index,
        base=40,
        num_classes=10,
        batch_size=32,
        stage1_epochs=20,
        stage2_epochs=20,
        save_path="best_multitask_model.pt"
):

    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FrameDataset(train_index, train=True)
    val_ds   = FrameDataset(val_index, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiTaskModel(base=base, num_classes=num_classes).to(device)

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)


    # STAGE 1
    best_val = float("inf")

    print("\n===== STAGE 1 (Cls + Seg) =====")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=stage1_epochs,
        eta_min=1e-6
    )

    for epoch in range(stage1_epochs):

        model.train()

        train_bar = tqdm(train_loader, desc=f"S1 Epoch {epoch+1}")

        for batch in train_bar:

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            mask = batch["mask"].to(device)
            has_mask = batch["has_mask"].to(device)

            cls_logits, seg_logits, _ = model(x)

            L_cls = criterion_cls(cls_logits, y)
            L_seg = masked_seg_loss(seg_logits, mask, has_mask)

            total = L_cls + L_seg

            optimizer.zero_grad()

            total.backward()

            optimizer.step()

            train_bar.set_postfix(loss=float(total))

        scheduler.step()

        val_loss, val_acc, seg_iou, _ = validate(
            model,
            val_loader,
            device,
            criterion_cls,
            stage=1
        )

        print(
            f"\nVal Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"Seg IoU: {seg_iou:.4f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(model.state_dict(), save_path)

            print("Saved Best Model")


    # STAGE 2
    print("\n===== STAGE 2 (With BBox) =====")

    best_bbox = 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    BBOX_WEIGHT = 2.0

    for epoch in range(stage2_epochs):

        model.train()

        train_bar = tqdm(train_loader, desc=f"S2 Train {epoch+1}")

        for bidx, batch in enumerate(train_bar):

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            mask = batch["mask"].to(device)
            bbox_gt = batch["bbox"].to(device)

            has_mask = batch["has_mask"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            cls_logits, seg_logits, bbox_pred = model(x)

            L_cls = criterion_cls(cls_logits, y)
            L_seg = masked_seg_loss(seg_logits, mask, has_mask)
            L_bbox = masked_bbox_loss_with_iou(bbox_pred, bbox_gt, has_bbox)

            total = L_cls + L_seg + BBOX_WEIGHT * L_bbox

            optimizer.zero_grad(set_to_none=True)

            total.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step(epoch + bidx/len(train_loader))

            train_bar.set_postfix({
                "Loss": f"{total.item():.3f}",
                "BBox": f"{L_bbox.item():.3f}"
            })

        val_loss, val_acc, seg_iou, bbox_iou = validate(
            model,
            val_loader,
            device,
            criterion_cls,
            stage=2
        )

        print(
            f"\nVal Acc: {val_acc:.4f} | "
            f"Seg IoU: {seg_iou:.4f} | "
            f"BBox IoU: {bbox_iou:.4f}"
        )

        if bbox_iou > best_bbox:

            best_bbox = bbox_iou

            torch.save({
                "model": model.state_dict(),
                "best_bbox": best_bbox,
                "epoch": epoch
            }, save_path)

            print(f" Saved Improved Model | BBox IoU: {best_bbox:.4f}")

    print("\nTraining Finished.")

    return model


DATASET_ROOT = " " # Dataset Path

GESTURE_CLASSES = [
    "G01_call",
    "G02_dislike",
    "G03_like",
    "G04_ok",
    "G05_one",
    "G06_palm",
    "G07_peace",
    "G08_rock",
    "G09_stop",
    "G10_three",
]

label_map = {name: i for i, name in enumerate(GESTURE_CLASSES)}

clip_index = build_clip_index(DATASET_ROOT, label_map)
frame_index = build_frame_index(clip_index)


train_frames, val_frames = split_dataset(frame_index)

train_dataset = FrameDataset(train_frames)
val_dataset = FrameDataset(val_frames, train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


model = train_two_stage(
    train_index=train_frames,
    val_index=val_frames,
    base=40,
    num_classes=10,
    batch_size=16,
    stage1_epochs=20,
    stage2_epochs=20
)