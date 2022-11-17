import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from typing import Iterable


class MedicDataset(Dataset):
    def __init__(self, file_path: str, images_dir: str, tasks: Iterable, transform: transforms.Compose = None) -> None:
        super().__init__()
        self.file_path = file_path
        self.images_dir = images_dir
        self.transform = transform
        self.tasks = tasks

        # Load data from tsv file
        df = pd.read_csv(file_path, sep="\t", na_filter=False)
        self.image_paths = df["image_path"].tolist()
        self.labels = [df[task].tolist() for task in self.tasks]
        self.classes = []
        self.class_indices = []
        for label in self.labels:
            pass

    @staticmethod
    def _find_classes(classes):
        classes_set = set(classes)
        classes = list(classes_set)
        classes.sort()
        if classes[0] == "":
            class_idx = {classes[i]: i - 1 for i in range(len(classes))}
        else:
            class_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_idx