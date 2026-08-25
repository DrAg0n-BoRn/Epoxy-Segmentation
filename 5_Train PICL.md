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
from dragon.ML_datasetmaster import DragonDatasetSegmentation as ChosenDataset
from dragon.ML_trainer import DragonVisionTrainer as ChosenTrainer
from dragon.ML_models_vision import DragonDeepLabv3 as ChosenModel
from dragon.ML_configuration import (
    FormatMultiClassSegmentationMetrics as ChosenMetricsConfig, 
    FinalizeMultiClassSegmentation as ChosenFinalizer
)

from dragon.ML_configuration import DragonTrainingConfig
from dragon.ML_callbacks import DragonPatienceEarlyStopping, DragonPlateauScheduler
from dragon.ML_utilities import build_optimizer_params
from dragon.keys import TaskKeys
from torch.optim import AdamW

from paths import PM
from helpers.picl import PhysicsInformedVoidsLoss as ChosenPICL
```

```python
!yes | plotly_get_chrome
```

```python
TRAIN_ARTIFACTS_DIR = PM.picl / "1_0-0_05"
DATASET_MANIFEST_FILE = PM.dataset_manifest_file
```

## 1. Config

```python
train_config = DragonTrainingConfig(
    initial_learning_rate=0.002,
    batch_size=16,
    task = TaskKeys.MULTICLASS_SEGMENTATION,
    device = "cuda:0",
    finalized_filename = "segmentation_deeplabv3_PICL_resnet101_epoxy",
    
    weight_decay=0.001,
    early_stop_patience=15,
    scheduler_patience=3,
    improvement_delta=0.0010,
    scheduler_lr_factor=0.6,
    monitor_metric="Validation Loss",
    baseline_weight=1.0,
    physics_weight=0.05,
)
```

## 2. Make Datasets

```python
dataset = ChosenDataset.from_manifest(DATASET_MANIFEST_FILE)
```

## 3. Model and Trainer

```python
model = ChosenModel(num_classes=len(dataset.class_map),
                    in_channels=dataset.image_channels,
                    model_name="deeplabv3_resnet101")

# optimizer
optim_params = build_optimizer_params(model=model, weight_decay=train_config.weight_decay)
optimizer = AdamW(params=optim_params, lr=train_config.initial_learning_rate)

# Physics informed loss
picl_loss = ChosenPICL(baseline_weight=train_config.baseline_weight, # type: ignore
                       physics_weight=train_config.physics_weight,) # type: ignore

trainer = ChosenTrainer(model=model,
                        train_dataset=dataset.train_dataset,
                        validation_dataset=dataset.validation_dataset,
                        save_dir=TRAIN_ARTIFACTS_DIR,
                        kind=train_config.task,
                        optimizer=optimizer,
                        device=train_config.device,
                        early_stopping_callback=DragonPatienceEarlyStopping(patience=train_config.early_stop_patience, 
                                                                            monitor=train_config.monitor_metric,
                                                                            min_delta=train_config.improvement_delta), # type: ignore
                        lr_scheduler_callback=DragonPlateauScheduler(monitor=train_config.monitor_metric,
                                                                     patience=train_config.scheduler_patience,
                                                                     factor=train_config.scheduler_lr_factor),
                        criterion=picl_loss
                        )
```

## 4. Training

```python
trainer.fit(epochs=500, 
            batch_size=train_config.batch_size, 
            use_torch_compile=True)
```

## 5. Evaluation

```python
trainer.evaluate(model_checkpoint="best",
                test_data=dataset.test_dataset,
                val_format_configuration=ChosenMetricsConfig(heatmap_cmap="plasma", radar_line_color="mediumspringgreen", cm_cmap="cool"),
                test_format_configuration=ChosenMetricsConfig(heatmap_cmap="magma", radar_line_color="crimson", cm_cmap="winter")
                )
```

## 6. Save artifacts

```python
# Model artifacts
model.save_architecture(TRAIN_ARTIFACTS_DIR)

# Train log
trainer.save_training_log(train_config=train_config)
```

## 7. Finalize Deep Learning

```python
trainer.finalize_model_training(finalize_config=ChosenFinalizer(filename=train_config.finalized_filename,
                                                                class_map=dataset.class_map)
                                )
```
