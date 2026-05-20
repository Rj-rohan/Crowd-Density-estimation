"""
Train CSRNet on ShanghaiTech Part A or Part B.

Before training, generate density maps:
  python data/generate_density_maps.py --root ShanghaiTech --part A --split train_data
  python data/generate_density_maps.py --root ShanghaiTech --part A --split test_data

Dataset structure used:
  ShanghaiTech/part_A/
    train_data/images/processed_IMG_N.jpg
    train_data/maps/IMG_N.npy
    test_data/images/processed_IMG_N.jpg
    test_data/maps/IMG_N.npy
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.csrnet import CSRNet


class ShanghaiTechDataset(Dataset):
    def __init__(self, root, part="A", split="train_data", patch_size=256):
        base = os.path.join(root, f"part_{part}", split)
        self.img_paths = sorted(glob.glob(os.path.join(base, "images", "*.jpg")))
        self.map_dir = os.path.join(base, "maps")
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # processed_IMG_N.jpg  →  IMG_N.npy
        stem = os.path.splitext(os.path.basename(img_path))[0]  # processed_IMG_N
        n = stem.replace("processed_IMG_", "")
        map_path = os.path.join(self.map_dir, f"IMG_{n}.npy")

        img = Image.open(img_path).convert("RGB")
        density = np.load(map_path).astype(np.float32)

        # Random crop
        w, h = img.size
        ps = self.patch_size
        if w > ps and h > ps:
            x = np.random.randint(0, w - ps)
            y = np.random.randint(0, h - ps)
            img = img.crop((x, y, x + ps, y + ps))
            density = density[y:y + ps, x:x + ps]

        img = self.transform(img)
        # CSRNet downsamples 8x; scale density to preserve count
        density_t = torch.from_numpy(density).unsqueeze(0)
        density_ds = nn.functional.avg_pool2d(density_t, 8, stride=8) * 64
        return img, density_ds


def evaluate(model, loader, device):
    model.eval()
    mae, mse = 0, 0
    with torch.no_grad():
        for imgs, targets in loader:
            pred = model(imgs.to(device)).sum().item()
            gt = targets.sum().item()
            err = pred - gt
            mae += abs(err)
            mse += err ** 2
    n = len(loader)
    return mae / n, (mse / n) ** 0.5


def train(
    data_root="ShanghaiTech",
    part="A",
    epochs=100,
    lr=1e-5,
    batch_size=8,
    save_path="weights/csrnet_shanghaitech.pth",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on ShanghaiTech Part {part} | device={device}")

    model = CSRNet(load_weights=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        ShanghaiTechDataset(data_root, part, "train_data"),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        ShanghaiTechDataset(data_root, part, "test_data"),
        batch_size=1, shuffle=False, num_workers=2,
    )

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mae, mse = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | MAE: {mae:.2f} | MSE: {mse:.2f}")

        if mae < best_mae:
            best_mae = mae
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (MAE={mae:.2f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="A", choices=["A", "B"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    train(part=args.part, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
