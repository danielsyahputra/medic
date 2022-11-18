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
        self.class_indices = []

        # Transform class to number
        for label in self.labels:
            cls_idx = self._find_classes(label)
            self.class_indices.append(cls_idx)
        # class_indices: [{"little_or_none": 0, "mild": 1, "severe": 2},
        #                 {"informative": 0, "not_informative": 1},
        #                 {.....}, ....]
        # len(class_indices) == len(self.tasks)
        self.all_targets = []
        num_images = len(self.labels[0])
        for image_idx in range(num_images):
            targets = []
            for task_idx in range(len(self.tasks)):
                targets.append(self.class_indices[task_idx][self.labels[task_idx][image_idx]])
            self.all_targets.append(targets)

    @staticmethod
    def _find_classes(classes):
        classes_set = set(classes)
        classes = list(classes_set)
        classes.sort()
        if classes[0] == "":
            class_idx = {classes[i]: i - 1 for i in range(len(classes))}
        else:
            class_idx = {classes[i]: i for i in range(len(classes))}
        return class_idx