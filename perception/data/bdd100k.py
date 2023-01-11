import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd

class BDD100k(data.Dataset):
    def __init__(self, root: str = '../bdd100k/', transforms: transforms = None, training: bool = False):
        self.root = root
        self.training = training
        self.det_path = self.root + 'labels/det_20/det_train.json' if self.training else 'labels/det_20/det_val.json'
        self.detections = pd.read_json(self.det_path)
        self.lane_path = self.root + 'labels/lane/masks/train/' if self.training else 'labels/lane/masks/val/'
        self.drivable_path = self.root + 'labels/drivable/masks/train/' if self.training else 'labels/drivable/masks/val/'
        
    def __len__(self):
        return len(self.detections)

    def __getitem__(self, index):
        target = self.detections[index]