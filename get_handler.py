import torch

from dragon.ML_inference_vision import DragonSegmentationInference
from dragon.ML_models_vision import DragonDeepLabv3
from dragon.ML_utilities import DragonArtifactFinder

from paths import PM


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_handler(chosen_model: str) -> DragonSegmentationInference:
    chosen_model_parsed = get_model_name(chosen_model)
    
    match chosen_model_parsed:
        case "Dice":
            model_architecture_path = PM.deeplab_dice
        case "GeneralizedDice-Focal":
            model_architecture_path = PM.deeplab_gen_dice_focal
        case "Focal":
            model_architecture_path = PM.deeplab_focal
        case "Tversky":
            model_architecture_path = PM.deeplab_tversky
        case _:
            model_architecture_path = PM.picl / chosen_model_parsed
    
    if not model_architecture_path.exists():
        raise FileNotFoundError(f"Model architecture path '{model_architecture_path}' does not exist.")
        
    artifact_finder = DragonArtifactFinder(directory=model_architecture_path, load_scaler=False)
    
    model = DragonDeepLabv3.load_architecture(artifact_finder.model_architecture_path)
    
    inference_handler = DragonSegmentationInference(model=model,
                                                    state_dict=artifact_finder.weights_path,
                                                    transform_source=PM.transform_recipe_file,
                                                    device=DEVICE)
    return inference_handler


def get_model_name(chosen_model: str) -> str:
    chosen_model_lower = chosen_model.strip().lower()
    
    if chosen_model_lower in ["", "baseline", "dice"]:
        return "Dice"
    elif "generalized" in chosen_model_lower:
        return "GeneralizedDice-Focal"
    elif "focal" in chosen_model_lower:
        return "Focal"
    elif "tversky" in chosen_model_lower:
        return "Tversky"
    else:
        return chosen_model
