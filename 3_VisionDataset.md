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
from dragon.ML_datasetmaster import DragonDatasetSegmentation

from paths import PM
from helpers.constants import CLASS_MAP, IMAGE_WINDOW_SIZE
```

## 1. Make Dataset Class

```python
dataset = DragonDatasetSegmentation.from_folders(image_dir=PM.tiled_images,
                                                 mask_dir=PM.tiled_masks)
```

```python
dataset.split_data(val_size=0.1,
                   test_size=0.1, 
                   random_state=101)
```

```python
dataset.set_class_map(CLASS_MAP)
```

```python
# inspect the directory of images to determine transforms to apply
dataset.inspect_folder(directory=PM.tiled_images,
                       save_dir_log=PM.dataset)
```

```python
dataset.configure_transforms(resize_size=IMAGE_WINDOW_SIZE)
```

```python
# inspect the dataset
print(dataset)
```

```python
# check channels after transforms
dataset.image_channels
```

## 2. Save Manifest File and artifacts

```python
OUTPUT_MANIFEST_FILE = PM.dataset_manifest_file
OUTPUT_TRANSFORM_RECIPE_FILE = PM.transform_recipe_file
```

```python
dataset.save_class_map(save_dir=OUTPUT_MANIFEST_FILE.parent)
```

```python
dataset.save_transform_recipe(filepath=OUTPUT_TRANSFORM_RECIPE_FILE)
```

```python
dataset.save_dataset_manifest(save_dir=OUTPUT_MANIFEST_FILE.parent, filename=OUTPUT_MANIFEST_FILE.name)
```
