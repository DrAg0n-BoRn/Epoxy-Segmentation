from dragon.path_manager import DragonPathManager


# 1. Initialize the PathManager using this file as the anchor, adding base directories.
PM = DragonPathManager(
    anchor_file=__file__,
    base_directories=["helpers", "results", "backups", "data"]
)


# 2. Define directories and files.
### Base files
PM.images = PM.data / "images"
PM.masks = PM.data / "masks"
PM.raw_masks = PM.data / "raw-masks"

### Tiled dataset
PM.tiled = PM.data / "images_tiled"
PM.tiled_images = PM.tiled / "images"
PM.tiled_masks = PM.tiled / "masks"

### VisionDataset
PM.dataset = PM.results / "3 VisionDataset"
PM.dataset_manifest_file = PM.dataset / "vision_dataset_manifest.json"
PM.transform_recipe_file = PM.dataset / "transform_recipe.json"

### Training
PM.segmentation_fcn = PM.results / "4 Segmentation FCN"

PM.segmentation_deeplab = PM.results / "4 Segmentation DeepLab"
PM.deeplab_dice = PM.segmentation_deeplab / "Dice"
PM.deeplab_focal = PM.segmentation_deeplab / "Focal"
PM.deeplab_gen_dice_focal = PM.segmentation_deeplab / "GeneralizedDice-Focal"
PM.deeplab_tversky = PM.segmentation_deeplab / "Tversky"

### PICL
PM.picl = PM.results / "5 PICL"

### Inference
PM.tiled_inference = PM.data / "images_inference_tiled"
PM.reconstructed_inference = PM.results / "6 Predicted Masks"

### Physical Properties
PM.properties = PM.results / "7 Physical Properties"
PM.properties_mask = PM.properties / "Masks"

### Statistics
PM.statistics = PM.results / "8 Statistics"


# 3. Make directories and check status
PM.make_dirs()

if __name__ == "__main__":
    PM.status()
