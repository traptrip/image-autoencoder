from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, root: str, stage: str, transform=None) -> None:
        super().__init__()
        self.root = Path(root)
        self.stage = stage
        assert stage in ["train", "val"]

        train = pd.read_csv(self.root / f"LOC_train_solution.csv")
        train.loc[:, "stage"] = "train"
        val = pd.read_csv(self.root / f"LOC_val_solution.csv")
        val.loc[:, "stage"] = "val"
        self.meta = pd.concat([train, val], axis=0).reset_index(drop=True)

        # get classes
        self.meta["target"] = [t.split()[0] for t in self.meta["PredictionString"]]
        self.label2id = {
            label: i for i, label in enumerate(sorted(self.meta.target.unique()))
        }

        # leave only N images per class
        filtered_meta = []
        for s in ["train", "val"]:
            n = 5 if s == "val" else 10
            m = self.meta.loc[self.meta.stage == s]
            for t in m.target.unique():
                t_meta = m.loc[m.target == t]
                filtered_meta.append(t_meta.iloc[:n])

        self.meta = pd.concat(filtered_meta, axis=0, ignore_index=True)

        # get stage metadata
        self.meta = self.meta.loc[self.meta.stage == stage].reset_index(drop=True)
        self.paths = [
            self.root / "ILSVRC/Data/CLS-LOC" / stage / (p.split("_")[0] if stage == "train" else ".") / f"{p}.JPEG"
            for p in self.meta["ImageId"]
        ]
        self.targets = [self.label2id[t] for t in self.meta.target]

        self.transform = transform

    def __getitem__(self, idx) -> Any:
        image = Image.open(self.paths[idx]).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.paths)
