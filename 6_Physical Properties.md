---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
from ml_tools.ML_inference_vision import DragonSegmentationInference
from ml_tools.ML_models_vision import DragonDeepLabv3
from ml_tools.path_manager import list_subdirectories
from ml_tools.IO_tools import save_json

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

from paths import PM
from helpers.constants import CLASS_FIBER_EPOXY_MATRIX, CLASS_POLYMER_COATING, CLASS_VOIDS
```

## Initialize Inference Engine

```python
model = DragonDeepLabv3.load_architecture(PM.segmentation_deeplab)
    
inference_handler = DragonSegmentationInference(model=model,
                                                state_dict= PM.segmentation_deeplab / "segmentation_deeplabv3_resnet101_epoxy.pth",
                                                transform_source=PM.transform_recipe,
                                                device="cuda:0")
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
        plt.figure(figsize=(10, 6))
        
        keys = list(data.keys())
        values = list(data.values())
        sns.barplot(x=keys, y=values, hue=keys, palette="viridis", legend=False)
        
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

## Polymer Coating Volume Fraction Calculation

```python
polymer_coating_volume_fraction: dict[str, float] = {}

for img_name, pixel_counts in all_images.items():
    current_volume_fraction = pixel_counts[CLASS_POLYMER_COATING] / (pixel_counts[CLASS_FIBER_EPOXY_MATRIX] + pixel_counts[CLASS_VOIDS] + pixel_counts[CLASS_POLYMER_COATING])
    polymer_coating_volume_fraction[img_name] = current_volume_fraction
```

```python
# visualize the polymer coating volume fraction results
plot_properties(data=polymer_coating_volume_fraction,
                ylabel="Polymer Coating Volume Fraction",
                filename="polymer_coating_volume_fraction.svg")
```

```python
# save the core matrix porosity results to a JSON file
save_json(data=core_matrix_porosity,
          directory=PM.properties_dir, 
          filename="polymer_coating_volume_fraction.json")
```
