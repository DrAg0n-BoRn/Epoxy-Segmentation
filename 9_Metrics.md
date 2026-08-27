---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: epoxy-segmentation (3.12.14)
    language: python
    name: python3
---

```python
from dragon.path_manager import list_subdirectories
from dragon.utilities import save_dataframe_filename

from paths import PM
from helpers.get_metrics import get_metrics, save_metrics_plot, save_combined_metrics_plot
```

## Get trained models metrics

```python
base_models_dict = list_subdirectories(PM.segmentation_deeplab)

pilf_models_dict = list_subdirectories(PM.picl)

all_models_dict = base_models_dict | pilf_models_dict
```

## Merge metrics

```python
df_test, df_validation = get_metrics(all_models_dict)
```

```python
df_test
```

```python
df_validation
```

```python
# save csv files
save_dataframe_filename(df=df_test, save_dir=PM.all_metrics, filename="all_test_metrics")
save_dataframe_filename(df=df_validation, save_dir=PM.all_metrics, filename="all_validation_metrics")
```

## Plot metrics

```python
save_metrics_plot(df=df_test,
                  save_dir=PM.all_metrics,
                  filename="test_metrics")
```

```python
save_metrics_plot(df=df_validation,
                  save_dir=PM.all_metrics,
                  filename="validation_metrics")
```

```python
save_combined_metrics_plot(df_wide=df_test,
                           save_dir=PM.all_metrics,
                           filename="combined_test_metrics")
```

```python
save_combined_metrics_plot(df_wide=df_validation,
                           save_dir=PM.all_metrics,
                           filename="combined_validation_metrics")
```
