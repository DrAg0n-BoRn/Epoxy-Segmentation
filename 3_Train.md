---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: epoxy-segmentation (3.12.3.final.0)
    language: python
    name: python3
---

```python
from ml_tools.ML_datasetmaster import DragonDatasetSegmentation as ChosenDataset
from ml_tools.ML_trainer import DragonTrainer as ChosenTrainer
from ml_tools.ML_models_vision import DragonFCN as ChosenModel
from ml_tools.ML_configuration import (
    FormatMultiClassSegmentationMetrics as ChosenMetricsConfig, 
    FinalizeMultiClassSegmentation as ChosenFinalizer, 
)

from ml_tools.ML_configuration import DragonTrainingConfig
from ml_tools.ML_callbacks import DragonModelCheckpoint, DragonPatienceEarlyStopping, DragonPlateauScheduler
from ml_tools.ML_utilities import build_optimizer_params, inspect_model_architecture
from ml_tools.IO_tools import train_logger, save_json
from ml_tools.keys import TaskKeys
from torch.optim import AdamW

from paths import PM
from helpers.constants import CLASS_MAP
```

```python
!yes | plotly_get_chrome
```

```python
TRAIN_ARTIFACTS_DIR = PM.segmentation_fcn
```

## 1. Config

```python
train_config = DragonTrainingConfig(
    validation_size=0.1,
    test_size=0.1,
    initial_learning_rate=0.005,
    batch_size=3,
    task = TaskKeys.MULTICLASS_SEGMENTATION,
    device = "cuda:0",
    finalized_filename = "segmentation_fcn_epoxy",
    random_state=101,
    
    weight_decay=0.02,
    early_stop_patience=5,
    scheduler_patience=2,
    scheduler_lr_factor=0.6,
    monitor_metric="Validation Loss"
)
```

## 2. Make Datasets

```python
dataset = ChosenDataset.from_folders(image_dir=PM.tiled_images_dir, mask_dir=PM.tiled_masks_dir)
```

```python
dataset.set_class_map(CLASS_MAP)
dataset.split_data(val_size=train_config.validation_size, test_size=train_config.test_size, random_state=train_config.random_state)
```

```python
dataset.configure_transforms(resize_size=512, 
                             crop_size=None)
```

```python
train_ds, val_ds, test_ds = dataset.get_datasets()
```

## 3. Model and Trainer

```python
model = ChosenModel(num_classes=len(CLASS_MAP),
                    in_channels=3,
                    model_name="fcn_resnet50")


# optimizer
optim_params = build_optimizer_params(model=model, weight_decay=train_config.weight_decay)
optimizer = AdamW(params=optim_params, lr=train_config.initial_learning_rate)

trainer = ChosenTrainer(model=model,
                        train_dataset=train_ds,
                        validation_dataset=val_ds,
                        save_dir=TRAIN_ARTIFACTS_DIR,
                        kind=train_config.task,
                        optimizer=optimizer,
                        device=train_config.device,
                        checkpoint_callback=DragonModelCheckpoint(monitor=train_config.monitor_metric),
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor=train_config.monitor_metric),
                        lr_scheduler_callback=DragonPlateauScheduler(monitor=train_config.monitor_metric,
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),  
                        )
```

## 4. Training

```python
history = trainer.fit(epochs=1000, batch_size=train_config.batch_size)
```

## 5. Evaluation

```python
trainer.evaluate(model_checkpoint="best",
                test_data=test_ds,
                val_format_configuration=ChosenMetricsConfig(),
                test_format_configuration=ChosenMetricsConfig(heatmap_cmap="viridis", radar_line_color="orange", cm_cmap="BuPu")
                )
```

## 6. Save artifacts

```python
# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)
inspect_model_architecture(model=model, save_dir=TRAIN_ARTIFACTS_DIR)

# Save class map
save_json(data=dataset.class_map, directory=TRAIN_ARTIFACTS_DIR, filename="class_map.json")

# Train log
train_logger(train_config=train_config,
             model_parameters=None,
             train_history=history,
             save_directory=TRAIN_ARTIFACTS_DIR)
```

## 7. Finalize Deep Learning

```python
trainer.finalize_model_training(model_checkpoint='current',
                                finalize_config=ChosenFinalizer(filename=train_config.finalized_filename,
                                                                class_map=dataset.class_map)
                                )
```
