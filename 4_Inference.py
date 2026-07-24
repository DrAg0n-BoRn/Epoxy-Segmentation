from ml_tools.ML_inference_vision import DragonSegmentationInference
from ml_tools.ML_models_vision import DragonDeepLabv3
from ml_tools.ML_vision_transformers import reconstruct_mask_overlapped_tiles
from ml_tools.path_manager import list_subdirectories
from ml_tools.ML_utilities import DragonArtifactFinder

from paths import PM

from typing import Literal


def load_model(chosen_model: Literal["deeplabv3", "deeplabv3_picl_10", "deeplabv3_picl_50", "deeplabv3_picl_99"]) -> DragonSegmentationInference:
    if chosen_model == "deeplabv3":
        model_architecture_path = PM.segmentation_deeplab
    elif chosen_model == "deeplabv3_picl_10":
        model_architecture_path = PM.segmentation_deeplab_picl_10
    elif chosen_model == "deeplabv3_picl_50":
        model_architecture_path = PM.segmentation_deeplab_picl_50
    elif chosen_model == "deeplabv3_picl_99":
        model_architecture_path = PM.segmentation_deeplab_picl_99
    else:
        raise ValueError(f"Invalid chosen_model: {chosen_model}. Must be one of ['deeplabv3', 'deeplabv3_picl_10', 'deeplabv3_picl_50', 'deeplabv3_picl_99']")
    
    artifact_finder = DragonArtifactFinder(directory=model_architecture_path, load_scaler=False, load_schema=False)
    
    model = DragonDeepLabv3.load_architecture(artifact_finder.model_architecture_path) # type: ignore
    
    inference_handler = DragonSegmentationInference(model=model,
                                                    state_dict=artifact_finder.weights_path, # type: ignore
                                                    transform_source=PM.transform_recipe,
                                                    device="cuda:0")
    return inference_handler


def make_predictions(inference_handler: DragonSegmentationInference):
    
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
    # Load the model and inference handler
    # TODO: Change the chosen_model parameter to select the desired model for inference
    inference_handler = load_model(chosen_model="deeplabv3_picl_99")
    
    # Make predictions on the tiled images and reconstruct the masks
    make_predictions(inference_handler=inference_handler)

