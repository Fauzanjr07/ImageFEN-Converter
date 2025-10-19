import json
import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


DEFAULT_CLASSES = [
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]


def save_classes_json(classes: List[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)


def load_classes_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["classes"]


class SquareFolderDataset(Dataset):
    """
    Simple ImageFolder-like dataset for single-square images.
    Expects directory structure:
        root/
           class_0/
             img1.jpg
             img2.png
           class_1/
             ...
    """

    def __init__(self, root: str, classes: List[str] = None, img_size: int = 224):
        self.root = root
        if classes is None:
            # infer from directories
            classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            classes = sorted(classes)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for c in self.classes:
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((os.path.join(cdir, fn), self.class_to_idx[c]))

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
