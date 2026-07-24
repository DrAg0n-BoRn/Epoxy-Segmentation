---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: epoxy-segmentation (3.12.12)
    language: python
    name: python3
---

```python
from ml_tools.ML_inference_vision import DragonSegmentationInference
from ml_tools.ML_models_vision import DragonDeepLabv3
from ml_tools.path_manager import list_subdirectories
from ml_tools.IO_tools import save_json
from ml_tools.ML_utilities import DragonArtifactFinder

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Literal

from paths import PM
from helpers.constants import CLASS_FIBER_EPOXY_MATRIX, CLASS_POLYMER_COATING, CLASS_VOIDS
```

## Initialize Inference Engine

```python
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
```

```python
inference_handler = load_model(chosen_model="deeplabv3_picl_99")
```

## Get pixel count per class

```python
# list all subdirectories in the tiled inference directory
all_subdirs = list_subdirectories(root_dir=PM.tiled_inference_dir)
```

```python
all_images: dict[str, dict[str, int]] = {}

for subdir_name, subdir_path in all_subdirs.items():
    pixel_count_per_class = inference_handler.predict_count_pixels_from_tiled_directory(directory_path=subdir_path, verbose=2)
    all_images[subdir_name] = pixel_count_per_class
```

```python
## helper visualization function to plot properties
def plot_properties(data: dict[str, Union[int, float]], ylabel: str, filename: str):
    """
    Generates a bar plot for the provided properties and saves it to disk.
    """
    with plt.rc_context({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14}):
        plt.figure(figsize=(12, 6))
        
        # Sort the dictionary by values in descending order
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
        keys = list(sorted_data.keys())
        values = list(sorted_data.values())
        sns.barplot(x=keys, y=values, hue=keys, palette="husl", legend=False)
        
        plt.title('')
        plt.xlabel('')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        
        sns.despine()
        
        plt.tight_layout()
        plt.savefig(PM.properties_dir / filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
```

## Core Matrix Porosity Calculation

```python
core_matrix_porosity: dict[str, float] = {}

for img_name, pixel_counts in all_images.items():
    current_porosity = pixel_counts[CLASS_VOIDS] / (pixel_counts[CLASS_FIBER_EPOXY_MATRIX] + pixel_counts[CLASS_VOIDS])
    core_matrix_porosity[img_name] = current_porosity
```

```python
# visualize the core matrix porosity results
plot_properties(data=core_matrix_porosity,
                ylabel="Core Matrix Porosity",
                filename="core_matrix_porosity.svg")
```

```python
# save the core matrix porosity results to a JSON file
save_json(data=core_matrix_porosity, 
          directory=PM.properties_dir, 
          filename="core_matrix_porosity.json")
```

## Polymer Coating Area Fraction Calculation

```python
polymer_coating_area_fraction: dict[str, float] = {}

for img_name, pixel_counts in all_images.items():
    current_area_fraction = pixel_counts[CLASS_POLYMER_COATING] / (pixel_counts[CLASS_FIBER_EPOXY_MATRIX] + pixel_counts[CLASS_VOIDS] + pixel_counts[CLASS_POLYMER_COATING])
    polymer_coating_area_fraction[img_name] = current_area_fraction
```

```python
# visualize the polymer coating area fraction results
plot_properties(data=polymer_coating_area_fraction,
                ylabel="Polymer Coating Area Fraction",
                filename="polymer_coating_area_fraction.svg")
```

```python
# save the polymer coating area fraction results to a JSON file
save_json(data=polymer_coating_area_fraction,
          directory=PM.properties_dir, 
          filename="polymer_coating_area_fraction.json")
```
