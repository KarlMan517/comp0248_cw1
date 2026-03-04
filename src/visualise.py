from utils import visualize_joint
from dataloader import build_clip_index, build_frame_index, FrameDataset
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

TEST_DATASET = "final_test" # Correct to the Test set path

test_clip_index = build_clip_index(TEST_DATASET, label_map)

test_frame_index = build_frame_index(test_clip_index)

test_dataset = FrameDataset(test_frame_index, train=False)

model = MultiTaskModel(base=40, num_classes=10).to(device)


visualize_joint(
    model,
    test_dataset,
    device,
    GESTURE_CLASSES,
    weight_path="./weights/best_multitask_model.pt",
    num_samples=5
)