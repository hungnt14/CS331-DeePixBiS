import argparse
import pandas as pd
import torch
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
)
from torch.utils.data import DataLoader

from utils.dataloader import PixWiseDataset
from models.model import DeePixBiS
from models.loss import PixWiseBCELoss
from utils.metrics import test_accuracy, test_loss

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="public_test_1_median_frame", help="exp name"
    )
    parser.add_argument(
        "--test_csv_path", default="./data/public_test_1_median_frame/public_test_1_median_frame.csv", help="csv data path"
    )
    parser.add_argument(
        "--weight_path",
        default="./weights/DeePixBiS_train_median_frame.pth",
        help="weight path",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="batch size"
    )
    return parser.parse_args()

def test(args):
    print("Exp: ", args.exp_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for testing")
    model = DeePixBiS()
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()
    model = model.to(device)
    loss_fn = PixWiseBCELoss()

    test_tfms = Compose(
        [Resize([224, 224]), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    dataset_df = pd.read_csv(args.test_csv_path)

    test_dataset = PixWiseDataset(dataset_df, transform=test_tfms)
    test_ds = test_dataset.dataset()

    batch_size = args.batch_size
    test_dl = DataLoader(
        test_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    test_acc = test_accuracy(model, test_dl)
    test_los = test_loss(model, test_dl, loss_fn)
    print(f"Test Accuracy : {test_acc}  Test Loss : {test_los}")

if __name__ == "__main__":
    args = get_parser()
    test(args)
