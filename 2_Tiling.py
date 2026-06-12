from ml_tools.ML_vision_transformers import make_tiled_dataset

from helpers.constants import CLASS_MAP, CLASS_BACKGROUND
from paths import PM


if __name__ == "__main__":
    
    #NOTE: Masks should be renamed to match their corresponding image files.
    
    make_tiled_dataset(input_dir=PM.images_dir,
                       mask_dir=PM.masks_dir,
                       window_size=512,
                       stride=0.8,
                       drop_empty_masks_by_value=CLASS_MAP[CLASS_BACKGROUND])

