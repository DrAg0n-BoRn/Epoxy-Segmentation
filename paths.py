from ml_tools.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "results", "backups", "data"]
)

# 2. Define directories and files.
### Base files
PM.images_dir = PM.data / "images"
PM.masks_dir = PM.data / "masks"
PM.raw_masks_dir = PM.data / "raw-masks"

### Tiled dataset
PM.tiled = PM.data / "images_tiled"
PM.tiled_images_dir = PM.tiled / "images"
PM.tiled_masks_dir = PM.tiled / "masks"

### Training
PM.segmentation_fcn = PM.results / "Segmentation FCN"



# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
