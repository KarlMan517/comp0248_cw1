import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from model import *
from tqdm import tqdm
import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score


# Reproducibility
def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# Depth Utilities
def normalize_depth(depth):

    depth = np.clip(depth, 0, 2000)

    depth = depth / 2000.0

    return depth


# Full Evaluation
def full_evaluation(
    model,
    loader,
    device,
    class_names,
    weight_path=None,
    show_confusion=True
):

    # Load Weight
    if weight_path is not None:

        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        print(" Model loaded from:", weight_path)

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    seg_ious = []
    dice_scores = []

    bbox_ious = []
    detection_hits = []


    # Evaluation Loop
    with torch.no_grad():

        for batch in tqdm(loader, desc="Evaluating"):

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            mask = batch["mask"].to(device)
            bbox = batch["bbox"].to(device)

            has_mask = batch["has_mask"].to(device)
            has_bbox = batch["has_bbox"].to(device)

            cls_logits, seg_logits, bbox_pred = model(images)


            # Classification
            preds = cls_logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


            # Segmentation Metrics

            probs = torch.sigmoid(seg_logits)

            seg_pred = (probs > 0.5).float()

            inter = (seg_pred * mask).sum(dim=(1,2,3))

            union = (seg_pred + mask - seg_pred*mask).sum(dim=(1,2,3)) + 1e-6

            iou = inter / union

            valid_mask = has_mask > 0.5

            if valid_mask.any():

                seg_ious.extend(iou[valid_mask].cpu().tolist())

                dice = (2 * inter) / (
                    (seg_pred + mask).sum(dim=(1,2,3)) + 1e-6
                )

                dice_scores.extend(dice[valid_mask].cpu().tolist())


            # Bounding Box Metrics

            valid_bbox = has_bbox > 0.5

            if valid_bbox.any():

                pred = bbox_pred[valid_bbox]
                gt = bbox[valid_bbox]

                x1 = torch.max(pred[:,0], gt[:,0])
                y1 = torch.max(pred[:,1], gt[:,1])

                x2 = torch.min(pred[:,2], gt[:,2])
                y2 = torch.min(pred[:,3], gt[:,3])

                inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

                area_p = (pred[:,2] - pred[:,0]) * (pred[:,3] - pred[:,1])
                area_g = (gt[:,2] - gt[:,0]) * (gt[:,3] - gt[:,1])

                union = area_p + area_g - inter + 1e-6

                biou = inter / union

                bbox_ious.extend(biou.cpu().tolist())

                detection_hits.extend((biou >= 0.5).cpu().numpy())


    # Final Metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    mean_seg_iou = np.mean(seg_ious) if len(seg_ious) > 0 else 0

    mean_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0

    mean_bbox_iou = np.mean(bbox_ious) if len(bbox_ious) > 0 else 0

    detection_acc = np.mean(detection_hits) if len(detection_hits) > 0 else 0


    # Print Results
    print("\n==============================")
    print("      FINAL EVALUATION")
    print("==============================")

    print("\nClassification")
    print(f"Top-1 Accuracy      : {accuracy:.4f}")
    print(f"Macro F1 Score      : {macro_f1:.4f}")

    print("\nSegmentation")
    print(f"Mean IoU            : {mean_seg_iou:.4f}")
    print(f"Mean Dice           : {mean_dice:.4f}")

    print("\nDetection")
    print(f"Detection Acc@0.5   : {detection_acc:.4f}")
    print(f"Mean BBox IoU       : {mean_bbox_iou:.4f}")


    # Confusion Matrix
    if show_confusion:

        cm = confusion_matrix(all_labels, all_preds)

        cm = cm.astype(np.float32) / (
            cm.sum(axis=1, keepdims=True) + 1e-6
        )

        plt.figure(figsize=(8,6))

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.title("Confusion Matrix")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig("./results/confusion_matrix.png", dpi=300)
        print("Saved Confusion Matrix.png")


    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_seg_iou": mean_seg_iou,
        "mean_dice": mean_dice,
        "mean_bbox_iou": mean_bbox_iou,
        "detection_acc_05": detection_acc
    }



# Visualization Function
def visualize_joint(
    model,
    dataset,
    device,
    class_names,
    weight_path=None,
    num_samples=4,
    indices=None,
    seg_area_thresh=0.001
):


    if weight_path is not None:
        checkpoint = torch.load(weight_path, map_location=device)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        print("Model loaded from:", weight_path)

    model.to(device)
    model.eval()


    # Select Samples
    if indices is None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))

    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)


    for row, idx in enumerate(indices):

        sample = dataset[idx]

        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["mask"].cpu().numpy()[0]
        gt_bbox = sample["bbox"].cpu().numpy()
        gt_label = sample["label"].item()
        has_bbox = sample["has_bbox"].item()

        with torch.no_grad():
            cls_logits, seg_logits, bbox_pred = model(image)

        pred_label = cls_logits.argmax(dim=1).item()


        # Segmentation
        pred_mask = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
        pred_mask_bin = (pred_mask > 0.5).astype(np.float32)


        # BBox
        pred_bbox = bbox_pred.cpu().numpy()[0]
        pred_bbox = np.clip(pred_bbox, 0.0, 1.0)


        # RGB
        rgb = sample["image"][:3].permute(1, 2, 0).cpu().numpy()
        rgb = np.clip(rgb, 0, 1)


        # Depth raw (channel 3)
        depth_raw = sample["image"][3].cpu().numpy()

        # normalize for display
        depth_vis = depth_raw.copy()
        depth_vis = depth_vis - depth_vis.min()
        depth_vis = depth_vis / (depth_vis.max() + 1e-6)

        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)

        H, W, _ = rgb.shape

        def denorm_box(box):
            x1 = int(box[0] * W)
            y1 = int(box[1] * H)
            x2 = int(box[2] * W)
            y2 = int(box[3] * H)
            return np.clip(x1, 0, W-1), np.clip(y1, 0, H-1), \
                   np.clip(x2, 0, W-1), np.clip(y2, 0, H-1)

        gt_x1, gt_y1, gt_x2, gt_y2 = denorm_box(gt_bbox)
        pr_x1, pr_y1, pr_x2, pr_y2 = denorm_box(pred_bbox)



        # RGB
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("RGB")
        axes[row, 0].axis("off")


        # Depth
        axes[row, 1].imshow(depth_vis)
        axes[row, 1].set_title("Depth (Raw)")
        axes[row, 1].axis("off")


        # Ground Truth
        gt_vis = (rgb * 255).astype(np.uint8).copy()

        if has_bbox:
            cv2.rectangle(gt_vis, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)

        axes[row, 2].imshow(gt_vis)
        axes[row, 2].imshow(gt_mask, alpha=0.4, cmap="Greens")
        axes[row, 2].set_title(f"GT: {class_names[gt_label]}")
        axes[row, 2].axis("off")


        # Prediction
        pr_vis = (rgb * 255).astype(np.uint8).copy()
        mask_area_ratio = pred_mask_bin.sum() / (H * W)

        if mask_area_ratio >= seg_area_thresh:
            cv2.rectangle(pr_vis, (pr_x1, pr_y1), (pr_x2, pr_y2), (255, 0, 0), 2)
            title_text = f"Pred: {class_names[pred_label]}"
        else:
            title_text = "No Segmentation"

        axes[row, 3].imshow(pr_vis)
        axes[row, 3].imshow(pred_mask_bin, alpha=0.4, cmap="Purples")
        axes[row, 3].set_title(title_text)
        axes[row, 3].axis("off")

    plt.savefig("./results/vis_output.png", dpi=300)
    print("Saved visualization to vis_output.png")