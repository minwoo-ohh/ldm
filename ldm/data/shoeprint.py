import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ShoeprintBase(Dataset):
    def __init__(self, data_root, size=None, interpolation="bicubic", flip_p=0.5):
        self.data_root = data_root
        self.sole_root = os.path.join(data_root, "sole")
        self.image_root = os.path.join(data_root, "footprint")
        self.paths = sorted([p for p in os.listdir(self.image_root)
                             if os.path.isfile(os.path.join(self.image_root, p))])
        self._length = len(self.paths)
        self.size = size
        self.interpolation = {"linear": Image.LINEAR,
                              "bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS}[interpolation]
        self.flip_p = flip_p

    def __len__(self):
        return self._length

    def _load_image(self, path):
        return Image.open(path)

    def __getitem__(self, i):
        fname = self.paths[i]
        img = self._load_image(os.path.join(self.image_root, fname)).convert("RGB")
        sole = self._load_image(os.path.join(self.sole_root, fname)).convert("L")

        img = np.array(img).astype(np.uint8)
        sole = np.array(sole).astype(np.uint8)

        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        sole = sole[(h - crop) // 2:(h + crop) // 2,
                    (w - crop) // 2:(w + crop) // 2]

        img = Image.fromarray(img)
        sole = Image.fromarray(sole)
        if self.size is not None:
            img = img.resize((self.size, self.size), resample=self.interpolation)
            sole = sole.resize((self.size, self.size), resample=self.interpolation)

        if random.random() < self.flip_p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            sole = sole.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.array(img).astype(np.uint8)
        sole = np.array(sole).astype(np.uint8)

        example = {
            "image": (img / 127.5 - 1.0).astype(np.float32),
            "sole": (sole / 127.5 - 1.0).astype(np.float32),
        }
        return example


class ShoeprintTrain(ShoeprintBase):
    def __init__(self, data_root="data/shoeprint", size=None, flip_p=0.5, **kwargs):
        super().__init__(data_root=data_root, size=size, flip_p=flip_p, **kwargs)


class ShoeprintValidation(ShoeprintBase):
    def __init__(self, data_root="data/shoeprint", size=None, flip_p=0.0, **kwargs):
        super().__init__(data_root=data_root, size=size, flip_p=flip_p, **kwargs)
