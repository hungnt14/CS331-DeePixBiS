import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    Normalize,
    Resize,
    ColorJitter,
)
from torch.utils.data import DataLoader

from utils.dataloader import PixWiseDataset
from models.model import DeePixBiS
from models.loss import PixWiseBCELoss
from trainer import Trainer

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="train_median_frames", help="exp name"
    )
    parser.add_argument(
        "--train_csv_path", default="./data/train_median_frame/train_median_frame.csv", help="csv data path"
    )
    parser.add_argument(
        "--n_epochs", default=5, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="batch size"
    )    
    parser.add_argument(
        "--train_ratio", default=0.8, type=float, help="train ratio"
    )
    parser.add_argument(
        "--saved_root",
        default="./weights/",
        help="weight path to store model",
    )
    return parser.parse_args()

def train(args):
    print("Exp: ", args.exp_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training")
    model = DeePixBiS()
    model.eval()
    loss_fn = PixWiseBCELoss()

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

    train_tfms = Compose(
        [
            Resize([224, 224]),
            RandomHorizontalFlip(0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    test_tfms = Compose(
        [Resize([224, 224]), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    dataset_df = pd.read_csv(args.train_csv_path)
    msk = np.random.rand(len(dataset_df)) < args.train_ratio
    train_df = dataset_df[msk]
    val_df = dataset_df[~msk]

    train_dataset = PixWiseDataset(train_df, transform=train_tfms)
    train_ds = train_dataset.dataset()

    val_dataset = PixWiseDataset(val_df, transform=test_tfms)
    val_ds = val_dataset.dataset()

    batch_size = args.batch_size
    train_dl = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    trainer = Trainer(train_dl, val_dl, model, args.n_epochs, opt, loss_fn, device)

    print("Training Beginning\n")
    trainer.fit()
    print("\nTraining Complete")
    torch.save(model.state_dict(), os.path.join(args.saved_root, f"./DeePixBiS_{args.exp_name}.pth"))
    print("Saved weight: ", os.path.join(args.saved_root, f"DeePixBiS_{args.exp_name}.pth"))

if __name__ == "__main__":
    args = get_parser()
    train(args)
