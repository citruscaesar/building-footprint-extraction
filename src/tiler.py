from pathlib import Path
import numpy as np
import rasterio as rio

class SceneTiler():
    def __init__(self, src_dir, tgt_dir, scene_files, tile_shape = (1024, 1024)):
        self.src_dir:Path = src_dir
        self.tgt_dir:Path = tgt_dir
        self.tile_shape = tile_shape
        self.scene_files = scene_files

        self._get_image_params()
        #self.tile()

    def tile(self):
        for filename in self.scene_files:
            path = self.src_dir / f"{filename}.tif"
            self._tile(path)

    def _load_raster(self, path):
        with rio.open(path) as raster:
            return raster.read()
    
    def _get_image_params(self):
        image = self._load_raster(self.src_dir / f"{self.scene_files[0]}.tif")
        self.scene_bands = image.shape[0]
        self.scene_shape = image.shape[1:]

    def _get_num_windows(self):
        (num_windows_x, remainder_x) = divmod(self.scene_shape[0], self.tile_shape[0])
        (num_windows_y, remainder_y) =  divmod(self.scene_shape[1], self.tile_shape[1])

        # If remainder is 0
        if not remainder_x:
            num_windows_x -= 1
        if not remainder_y:
            num_windows_y -= 1

        return num_windows_x, num_windows_y

    def _get_window_coords(self):

        #TODO: Fix Window creation logic to avoid overlaps
        #Overlaps means tiler is called on the same coordinates multiple times

        num_windows_x, num_windows_y = self._get_num_windows()
        for y_incr in range(num_windows_y+1):
            for x_incr in range(num_windows_x+1):
                x_coordinate = x_incr * self.tile_shape[0]
                y_coordinate = y_incr * self.tile_shape[1]
                yield (x_coordinate, y_coordinate)
                if x_incr == num_windows_x:
                    yield (self.scene_shape[0] - self.tile_shape[0], y_coordinate)
                if y_incr == num_windows_y:
                    yield (x_coordinate, self.scene_shape[1] - self.tile_shape[1])

    def _tile(self, scene_path) -> None:
        with rio.open(scene_path) as scene:
            scene_name = scene.name.split('/')[-1].split('.')[0]
            profile = scene.profile
            for row, col in self._get_window_coords():
                window = rio.windows.Window(row, col, self.tile_shape[0], self.tile_shape[1])        
                transform = scene.window_transform(window)
                profile.update({
                    'width': self.tile_shape[0],
                    'height': self.tile_shape[1],
                    'transform': transform
                })
                tile_path = self.tgt_dir / f"{scene_name}-{row}x{col}.tif"
                with rio.open(tile_path.as_posix(), 'w', **profile) as tile:
                    tile.write(scene.read(window = window))