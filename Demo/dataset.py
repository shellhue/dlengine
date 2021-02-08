
import torch
from PIL import Image


class BaseClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, classes, transform=None):
        pass

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        cls = img_path.split("/")[-2]
        cls = self.decode(cls)
        cls_id = self.class_to_id[cls]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, cls_id