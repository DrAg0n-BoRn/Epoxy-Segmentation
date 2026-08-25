from dragon.ML_vision_utilities import merge_masks_with_inferred_class, inspect_folder

from helpers.constants import CLASS_MAP, CLASS_FIBER_EPOXY_MATRIX
from paths import PM


if __name__ == "__main__":
    merge_masks_with_inferred_class(input_dir=PM.raw_masks,
                                    output_dir=PM.masks,
                                    class_map=CLASS_MAP,
                                    inferred_class_name=CLASS_FIBER_EPOXY_MATRIX)
    
    inspect_folder(PM.masks)
