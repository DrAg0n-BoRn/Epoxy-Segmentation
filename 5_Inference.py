from ml_tools.ML_inference_vision import DragonSegmentationInference
from ml_tools.ML_models_vision import DragonDeepLabv3
from ml_tools.ML_vision_transformers import reconstruct_mask_overlapped_tiles
from ml_tools.path_manager import list_subdirectories

from paths import PM


def main():
    
    model = DragonDeepLabv3.load_architecture(PM.segmentation_deeplab)
    
    inference_handler = DragonSegmentationInference(model=model,
                                                    state_dict= PM.segmentation_deeplab / "segmentation_deeplabv3_resnet101_epoxy.pth",
                                                    transform_source=PM.transform_recipe,
                                                    device="cuda:0")
    
    # clean overlapped files from previous runs
    inference_handler.clear_overlapped_images(PM.tiled_inference_dir)
    
    # list all subdirectories in the tiled inference directory
    all_subdirs = list_subdirectories(root_dir=PM.tiled_inference_dir)
    
    for _subdir_name, subdir_path in all_subdirs.items():
        # run inference on the tiled images in the subdirectory to generate overlapping predicted masks
        inference_handler.predict_from_directory(directory_path=subdir_path, valid_extensions=[".png"], verbose=2)
        
        # reconstruct the overlapping tiles into a single mask
        reconstruct_mask_overlapped_tiles(input_dir=subdir_path, 
                                          output_dir=PM.reconstructed_inference_dir,
                                          verbose=2)


if __name__ == "__main__":
    main()
