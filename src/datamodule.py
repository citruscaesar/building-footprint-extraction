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

class InriaDataset(Dataset):
    def __init__(self, scenes_dir:Path, masks_dir:Path, subset_scenes = None, preprocess_fn = None, scene_transforms = None, mask_transforms = None):
        super().__init__()

        self.scenes_dir:Path = scenes_dir 
        self.masks_dir:Path = masks_dir 

        #Scenes
        if subset_scenes is not None:
            self.scenes:list = [self.scenes_dir / f"{x}.tif" for x in subset_scenes]
            self.masks:list = [self.masks_dir / f"{x}.tif" for x in subset_scenes]
        else:
            self.scenes:list = list(self.scenes_dir.iterdir())
            self.masks:list = list(self.masks_dir.iterdir())

        #Transforms
        self.preprocess_fn = preprocess_fn
        self.scene_transforms = scene_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self._load_raster(self.scenes[idx])
        mask = self._load_raster(self.masks[idx])

        if self.preprocess_fn is not None:
            scene = self.preprocess_fn(scene.permute(2, 1, 0))
            scene = scene.permute(2, 0, 1).float()
        else:
            scene = (scene / 255.).float()

        mask = torch.clip(mask, 0., 1.).int()

        #Additional Transformations
        if self.scene_transforms is not None:
            scene = self.scene_transforms(scene)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        return scene, mask

    def _load_raster(self, path):
        with rio.open(path) as raster:
            return torch.from_numpy(raster.read())


class InriaTiledDataset(InriaDataset):
    def __init__(self, root, split = "train", preprocess_fn = None, scene_transforms = None, mask_transforms = None):
        
        self.root:Path = root

        assert split in ["train", "test"]
        self.split = split

        self.scenes_dir:Path = self.root / self.split / "scenes"
        self.masks_dir:Path = self.root / self.split / "masks"

        #Transforms
        self.preprocess_fn = preprocess_fn
        self.scene_transforms = scene_transforms
        self.mask_transforms = mask_transforms

        super().__init__(scenes_dir=self.scenes_dir, 
                         masks_dir=self.masks_dir, 
                         preprocess_fn=self.preprocess_fn,
                         scene_transforms=self.scene_transforms,
                         mask_transforms=self.mask_transforms)


class InriaDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root:Path, 
                 preprocess_fn = None, 
                 batch_size:int = 32, 
                 tile_shape = None, 
                 num_workers = 0,
                 scene_transforms = None,
                 mask_transforms = None):
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

        if tile_shape is None: 
            self.tiling_required = False
            #self.tile_shape = None
        else:
            assert type(tile_shape) == tuple
            assert len(tile_shape) == 2
            self.tile_shape = tile_shape 
            self.tiling_required = True 
            self.tiled_dir = self.root.parent / "tiled" / f"{self.tile_shape[0]}x{self.tile_shape[1]}"

        self.batch_size = batch_size 
        self.preprocess_fn = preprocess_fn
        self.num_workers = num_workers

        #Transforms
        self.scene_transforms = scene_transforms
        self.mask_transforms = mask_transforms
     
    def prepare_data(self):
        if self.tiling_required and not self._check_tiled_dirs():
            print("Tiled Directories Not Found")
            self._create_dirs()
            print(f"Created Tiled Directories")
            self._tile() 
            print(f"Created Tiles of Shape: {self.tile_shape}")

            InriaTiledDataset(self.tiled_dir)
            print("Dataset Initialization Successful")
        
        elif not self.tiling_required:
            print("Tiling Not Required")
            print("DataModule Will Return Entire Scenes")

            InriaDataset(self.src_dirs["scenes"], self.src_dirs["masks"])
            print("Dataset Initialization Successful")

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            assert self.tiling_required == True, "Tiling Disabled"
            dataset = InriaTiledDataset(root = self.tiled_dir, 
                                        split = "train", 
                                        preprocess_fn = self.preprocess_fn,
                                        scene_transforms = self.scene_transforms,
                                        mask_transforms=self.mask_transforms)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [.8, .2])

        elif stage == "test":
            assert self.tiling_required == True, "Tiling Disabled"
            self.test_dataset = InriaTiledDataset(root = self.tiled_dir, 
                                                  split = "test", 
                                                  preprocess_fn = self.preprocess_fn)

        elif stage == "predict":
            #assert self.tiling_required == False, "Tiling Enabled"
            self.pred_dataset = InriaDataset(scenes_dir = self.src_dirs["scenes"], 
                                             masks_dir = self.src_dirs["masks"], 
                                             subset_scenes = self.scene_split["test"], 
                                             preprocess_fn = self.preprocess_fn,
                                             scene_transforms=self.scene_transforms,
                                             mask_transforms=self.mask_transforms)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True, num_workers = self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, num_workers = self.num_workers)


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
    
    def _check_tiled_dirs(self) -> bool:
        for path in self._list_tiled_dirs():
            if not path.exists() or not path.is_dir():
                return False
        return True

    def _create_dirs(self) -> bool:
        for path in self._list_tiled_dirs():
            path.mkdir(parents = True, exist_ok = True)