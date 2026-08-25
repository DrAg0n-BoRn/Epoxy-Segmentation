import argparse

from dragon.ML_inference_vision import DragonSegmentationInference
from dragon.ML_vision_utilities import reconstruct_mask_overlapped_tiles
from dragon.path_manager import list_subdirectories

from paths import PM
from get_handler import get_handler, get_model_name


def make_predictions(inference_handler: DragonSegmentationInference, model_name: str = ""):    
    # list all subdirectories in the tiled inference directory
    all_subdirs = list_subdirectories(root_dir=PM.tiled_inference)
    
    # output dir
    model_name = get_model_name(model_name)
    
    output_path = PM.reconstructed_inference / model_name
    
    for _subdir_name, subdir_path in all_subdirs.items():
        # run inference on the tiled images in the subdirectory to generate overlapping predicted masks
        inference_handler.predict_from_directory(directory_path=subdir_path, valid_extensions=[".png"], verbose=2)
        
        # reconstruct the overlapping tiles into a single mask
        reconstruct_mask_overlapped_tiles(input_dir=subdir_path, 
                                          output_dir=output_path,
                                          verbose=2)
    
    # cleanup overlapped files 
    inference_handler.clear_overlapped_images(PM.tiled_inference)


if __name__ == "__main__":
    # Load the model and inference handler
    argument_parser = argparse.ArgumentParser(description="Run inference on tiled images and reconstruct masks.")
    argument_parser.add_argument("model", type=str, help="The name of the model to use for inference. Empty string or 'baseline' for baseline model.") #positional argument for the model name
    args = argument_parser.parse_args()
    
    # Get the inference handler for the specified model
    inference_handler = get_handler(chosen_model=args.model)
    
    # Make predictions on the tiled images and reconstruct the masks
    make_predictions(inference_handler=inference_handler, model_name=args.model)
