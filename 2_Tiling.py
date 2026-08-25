from dragon.ML_vision_utilities import make_tiled_dataset, make_tiled_inference

from helpers.constants import CLASS_MAP, CLASS_BACKGROUND, IMAGE_WINDOW_SIZE
from paths import PM


if __name__ == "__main__":
    
    #NOTE: Masks should be renamed to match their corresponding image files.
    
    make_tiled_dataset(input_dir=PM.images,
                       mask_dir=PM.masks,
                       window_size=IMAGE_WINDOW_SIZE,
                       ratio_strategy="shift",
                       stride=0.8,
                       drop_empty_masks_by_value=CLASS_MAP[CLASS_BACKGROUND])
    
    # inference tiling
    make_tiled_inference(input_dir=PM.images,
                        window_size=IMAGE_WINDOW_SIZE,
                        ratio_strategy="shift"
                        )
