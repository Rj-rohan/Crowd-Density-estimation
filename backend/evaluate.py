"""Evaluate CSRNet on ShanghaiTech test split. Reports MAE and MSE."""

import argparse
import torch
from torch.utils.data import DataLoader
from models.csrnet import CSRNet
from train_csrnet import ShanghaiTechDataset


def run_evaluation(weights="weights/csrnet_shanghaitech.pth", data_root="ShanghaiTech", part="A"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    loader = DataLoader(ShanghaiTechDataset(data_root, part, "test_data"), batch_size=1, shuffle=False)
    mae_sum, mse_sum, n = 0, 0, 0

    with torch.no_grad():
        for imgs, targets in loader:
            pred = model(imgs.to(device)).sum().item()
            gt = targets.sum().item()
            err = pred - gt
            mae_sum += abs(err)
            mse_sum += err ** 2
            n += 1

    mae = mae_sum / n
    mse = (mse_sum / n) ** 0.5
    print(f"ShanghaiTech Part {part} — test split ({n} samples)")
    print(f"  MAE : {mae:.2f}")
    print(f"  MSE : {mse:.2f}")
    return {"mae": mae, "mse": mse, "samples": n}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="A", choices=["A", "B"])
    parser.add_argument("--weights", default="weights/csrnet_shanghaitech.pth")
    args = parser.parse_args()
    run_evaluation(weights=args.weights, part=args.part)
