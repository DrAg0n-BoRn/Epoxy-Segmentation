from ml_tools.ML_vision_transformers import merge_masks_with_inferred_class, inspect_folder

from helpers.constants import CLASS_MAP, CLASS_FIBER_EPOXY_MATRIX
from paths import PM


if __name__ == "__main__":
    merge_masks_with_inferred_class(input_dir=PM.raw_masks_dir,
                                    output_dir=PM.masks_dir,
                                    class_map=CLASS_MAP,
                                    inferred_class_name=CLASS_FIBER_EPOXY_MATRIX)
    
    inspect_folder(PM.masks_dir)
