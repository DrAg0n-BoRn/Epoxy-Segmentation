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
from dragon.path_manager import list_subdirectories

from paths import PM
from get_handler import get_handler, get_model_name
from helpers.get_properties import save_relative_properties
```

```python
#TODO: Specify model name for the current run
CHOSEN_MODEL = "1_0-0_05"
```

## Initialize Inference Engine

```python
assert len(CHOSEN_MODEL) > 0, "Specify a model name for the current run."
inference_handler = get_handler(chosen_model=CHOSEN_MODEL)
```

```python
OUTPUT_DIR = PM.properties / get_model_name(chosen_model=CHOSEN_MODEL)
```

## Get pixel count per class

```python
# list all subdirectories in the tiled inference directory
all_subdirs = list_subdirectories(root_dir=PM.tiled_inference)
```

```python
all_images: dict[str, dict[str, int]] = {}

for subdir_name, subdir_path in all_subdirs.items():
    pixel_count_per_class = inference_handler.predict_count_pixels_from_tiled_directory(directory_path=subdir_path, verbose=2)
    all_images[subdir_name] = pixel_count_per_class
```

## Relative Matrix Porosity and Relative Polymer Coating Area Calculation

```python
save_relative_properties(images_pixel_counts=all_images, output_dir=OUTPUT_DIR)
```

## Mask properties calculation (run once)

```python
from dragon.ML_vision_utilities import count_mask_pixels_by_class

from helpers.constants import CLASS_MAP
```

```python
mask_pixel_counts = count_mask_pixels_by_class(directory=PM.masks, 
                                                class_map=CLASS_MAP, 
                                                verbose=2)
```

```python
save_relative_properties(images_pixel_counts=mask_pixel_counts, output_dir=PM.properties_mask)
```
