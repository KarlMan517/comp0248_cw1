import os
import random
from glob import glob

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from collections import defaultdict


# =========================================================
# Index Builders
# =========================================================

def build_clip_index(dataset_root, label_map):
    clip_index = []
    student_folders = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])

    print("Found students:", len(student_folders))
    print("Example student folders:", student_folders[:3])

    for student in student_folders:
        student_path = os.path.join(dataset_root, student)

        gesture_folders = sorted([g for g in os.listdir(student_path) if os.path.isdir(os.path.join(student_path, g))])

        # Skip if its not in the label list
        for gesture in gesture_folders:
            if gesture not in label_map:
                continue


            label = label_map[gesture]
            gesture_path = os.path.join(student_path, gesture)

            clip_folders = sorted([c for c in os.listdir(gesture_path) if os.path.isdir(os.path.join(gesture_path, c))])

            for clip_name in clip_folders:
                clip_path = os.path.join(gesture_path, clip_name)

                rgb_paths = sorted(glob(os.path.join(clip_path, "rgb", "*.png")))
                depth_paths = sorted(glob(os.path.join(clip_path, "depth_raw", "*.npy")))
                mask_paths = sorted(glob(os.path.join(clip_path, "annotation", "*.png")))

                # Build dictionary
                # frame_001.png = mask_path
                mask_dict = {os.path.basename(p): p for p in mask_paths}

                frames = []
                for rgb in rgb_paths:

                    fn = os.path.basename(rgb)
                    base_name = os.path.splitext(fn)[0]   # frame_001

                    depth = None
                    for d in depth_paths:
                        depth_name = os.path.splitext(os.path.basename(d))[0]
                        if depth_name == base_name:
                            depth = d
                            break

                    mask = mask_dict.get(fn, None)

                    frames.append({
                        "rgb": rgb,
                        "depth": depth,
                        "mask": mask
                    })

                clip_index.append({
                    "student_id": student,
                    "gesture": gesture,
                    "label": label,
                    "clip_dir": clip_path,
                    "frames": frames
                })

    return clip_index


def build_frame_index(clip_index):
    frame_index = []
    for clip in clip_index:
        for f in clip["frames"]:
            frame_index.append({
                "rgb": f["rgb"],
                "depth": f["depth"],
                "mask": f["mask"],
                "has_mask": 1 if f["mask"] is not None else 0,
                "label": clip["label"],
                "gesture": clip["gesture"],
                "student_id": clip["student_id"],
                "clip_dir": clip["clip_dir"]
            })

    return frame_index


def split_dataset(frame_index, val_ratio=0.2, seed=42):
    random.seed(seed)

    student_to_frames = defaultdict(list)
    for item in frame_index:
        student_to_frames[item["student_id"]].append(item)

    students = list(student_to_frames.keys())
    random.shuffle(students)

    split_idx = int(len(students)*(1-val_ratio))
    train_students = students[:split_idx]
    val_students = students[split_idx:]

    train_frames = []
    val_frames = []

    for s in train_students:
        train_frames.extend(student_to_frames[s])

    for s in val_students:
        val_frames.extend(student_to_frames[s])

    print("Train students:", len(train_students))
    print("Val students:", len(val_students))
    print("Train frames:", len(train_frames))
    print("Val frames:", len(val_frames))

    return train_frames, val_frames



# Depth Utilities
def load_depth(path):

    if path is None:
        return None

    return np.load(path).astype(np.float32)


def normalize_depth(depth):

    depth = np.clip(depth, 0, 2000)
    depth = depth / 2000.0

    return depth



# Dataset
class FrameDataset(Dataset):
    def __init__(self, frame_index, img_size=256, train=True):
        self.items = frame_index
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.items)

    def random_geometric(self, img, mask):
        H, W = mask.shape

        if random.random() < 0.3:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.3:
            angle = random.uniform(-8, 8)
            M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)

            img = cv2.warpAffine(
                img, M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            mask = cv2.warpAffine(
                mask, M, (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        return img, mask

    def random_color(self, img):
        rgb = img[:, :, :3]

        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.2)
            rgb = np.power(rgb, gamma)

        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)
            rgb *= alpha

        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, rgb.shape)
            rgb += noise

        rgb = np.clip(rgb, 0, 1)
        img[:, :, :3] = rgb
        return img

    def rgb_dropout(self, img):
        if random.random() < 0.15:
            img[:, :, :3] = 0
        return img

    def process_depth(self, depth):
        depth = cv2.resize(
            depth, (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )
        depth = normalize_depth(depth)

        sobelx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge = edge / (edge.max() + 1e-6)

        return depth, edge

    def __getitem__(self, idx):
        it = self.items[idx]

        # RGB
        rgb = cv2.imread(it["rgb"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        rgb = rgb.astype(np.float32) / 255.0

        # Depth
        depth = load_depth(it["depth"])
        if depth is None:
            depth_raw = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            depth_edge = np.zeros_like(depth_raw)
        else:
            depth_raw, depth_edge = self.process_depth(depth)

        depth_2ch = np.stack([depth_raw, depth_edge], axis=-1)
        img = np.concatenate([rgb, depth_2ch], axis=-1)  # (H,W,5)

        # Mask
        if it.get("mask", None) is not None:
            mask = cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            has_mask = 1
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            has_mask = 0

        # Augmentation
        if self.train:
            img, mask = self.random_geometric(img, mask)
            img = self.random_color(img)
            img = self.rgb_dropout(img)

        # Bounding Box
        PAD_RATIO = 0.02  # 2% padding

        if has_mask:
            ys, xs = np.where(mask > 0)

            if len(xs) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                H, W = mask.shape

                # Convert to normalized
                x1 /= W
                x2 /= W
                y1 /= H
                y2 /= H

                # Pad (relative to image size)
                pad_x = PAD_RATIO
                pad_y = PAD_RATIO

                x1 = max(0.0, x1 - pad_x)
                y1 = max(0.0, y1 - pad_y)
                x2 = min(1.0, x2 + pad_x)
                y2 = min(1.0, y2 + pad_y)

                bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                has_bbox = 1
            else:
                bbox = np.zeros(4, dtype=np.float32)
                has_bbox = 0
                has_mask = 0
        else:
            bbox = np.zeros(4, dtype=np.float32)
            has_bbox = 0

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()   # (5,H,W)
        mask = torch.from_numpy(mask)[None, ...].contiguous().float()       # (1,H,W)
        bbox = torch.from_numpy(bbox).contiguous().float()                  # (4,)

        return {
            "image": img,
            "mask": mask,
            "bbox": bbox,
            "has_mask": torch.tensor(has_mask, dtype=torch.float32),
            "has_bbox": torch.tensor(has_bbox, dtype=torch.float32),
            "label": torch.tensor(it["label"], dtype=torch.long),
        }