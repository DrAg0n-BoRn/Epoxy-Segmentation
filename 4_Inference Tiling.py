from ml_tools.ML_vision_transformers import make_tiled_inference

from helpers.constants import IMAGE_WINDOW_SIZE
from paths import PM


if __name__ == "__main__":
    
    make_tiled_inference(input_dir=PM.images_dir,
                         window_size=IMAGE_WINDOW_SIZE,
                         ratio_strategy="pad-black"
                         )

