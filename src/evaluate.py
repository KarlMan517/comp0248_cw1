from utils import full_evaluation
from dataloader import build_clip_index, build_frame_index, FrameDataset
from torch.utils.data import DataLoader
from model import *
import torch

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

TEST_DATASET = "final_test"

test_clip_index = build_clip_index(TEST_DATASET, label_map)

test_frame_index = build_frame_index(test_clip_index)

test_dataset = FrameDataset(test_frame_index, train=False)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

model = MultiTaskModel(base=40, num_classes=10).to(device)

results = full_evaluation(
    model,
    test_loader,
    device,
    GESTURE_CLASSES,
    weight_path="./weights/best_multitask_model.pt"
)