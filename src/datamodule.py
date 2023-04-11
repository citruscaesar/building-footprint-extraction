from pathlib import Path
import numpy as np
import rasterio as rio

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from .tiler import SceneTiler

'''
root -> tiled -> tile_shape -> train -> scene, mask
root -> tiled -> tile_shape -> test -> scene, mask
root -> images, gt
'''

class InariaDataset(Dataset):
    def __init__(self, root, split = "train", scene_transforms = None, mask_transforms = None):
        super().__init__()

        self.root:Path = root
        
        #Train, Test, Pretrain Splits
        assert split in ["train", "test"]
        self.split = split

        #self.scene_locations = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
        #self.train_scenes = [f"{loc}{idx}.tif" for idx in range(6, 37) for loc in self.scene_locations] 
        #self.test_scenes = [f"{loc}{idx}.tif" for idx in range(1, 6) for loc in self.scene_locations]

        #Important Paths
        #Tiled Directories
        self.scenes_dir:Path = self.root / self.split / "scenes"
        self.masks_dir:Path = self.root / self.split / "masks"

        #Tiled Images
        self.scenes:Path = list(self.scenes_dir.iterdir())
        self.masks:Path = list(self.masks_dir.iterdir())

        #Transforms
        self.scene_transforms = scene_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(list(self.scenes_dir.iterdir()))

    def __getitem__(self, idx):
        scene = self._load_raster(self.scenes[idx])
        mask = self._load_raster(self.masks[idx])

        if self.scene_transforms is not None:
            scene = self.scene_transforms(scene.to(torch.float))
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        mask = torch.clip(mask, 0., 1.).to(torch.int)
        scene = scene / 255.
        return scene, mask

    def _load_raster(self, path):
        with rio.open(path) as raster:
            return torch.from_numpy(raster.read())


class InariaDataModule(pl.LightningDataModule):
    def __init__(self, root:Path, batch_size:int = 32, tile_shape = (512, 512), num_workers = 16):
        super().__init__()
        #Paths
        self.root:Path = root
        self.src_dirs = {
            "scenes": self.root / "images",
            "masks": self.root / "gt" 
        }

        #Splits
        self.scene_locations = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
        self.scene_split = {
            "train": [f"{loc}{idx}" for idx in range(6, 37) for loc in self.scene_locations],
            "test": [f"{loc}{idx}" for idx in range(1, 6) for loc in self.scene_locations]
        }

        self.batch_size = batch_size 
        self.tile_shape = tile_shape 
        self.num_workers = num_workers

        self.tiled_dir = self.root.parent / "tiled" / f"{self.tile_shape[0]}x{self.tile_shape[1]}"
     
    def prepare_data(self):
        if not self._check_dirs():
            print("Tiled Directories Not Found")
            self._create_dirs()
            self._tile() 
        InariaDataset(self.tiled_dir)

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            dataset = InariaDataset(self.tiled_dir, split = "train")
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [.8, .2])
        elif stage == "test" or stage == "predict":
            self.test_dataset = InariaDataset(self.tiled_dir, split = "test")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers = self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)


    def _tile(self):
        for path in self._list_tiled_dirs():
            tiler = SceneTiler(self.src_dirs[path.name], 
                               path, 
                               self.scene_split[path.parent.name], 
                               self.tile_shape)
            tiler.tile()

    def _list_tiled_dirs(self) -> list:
        s = self.tile_shape
        tiled_dirs = list()
        for split in ["train", "test"]:
            for img_type in ["scenes", "masks"]:
                path = self.root.parent / "tiled" / f"{s[0]}x{s[1]}" / split / img_type
                tiled_dirs.append(path)
        return tuple(tiled_dirs)
    
    def _check_dirs(self) -> bool:
        for path in self._list_tiled_dirs():
            if not path.exists() or not path.is_dir():
                return False
        return True

    def _create_dirs(self) -> bool:
        for path in self._list_tiled_dirs():
            path.mkdir(parents = True, exist_ok = True)