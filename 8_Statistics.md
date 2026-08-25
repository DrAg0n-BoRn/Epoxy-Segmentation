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
from dragon.data_exploration import summarize_dataframe
from dragon.utilities import save_dataframe_filename

from paths import PM
from helpers.get_statistics import parse_strategies_to_df, plot_model_errors, get_relative_error_df, get_absolute_error_df
from helpers.constants import RELATIVE_COATING_AREA_FILENAME, RELATIVE_MATRIX_POROSITY_FILENAME
```

```python
# find all strategy subdirectories
strategy_dirs = list_subdirectories(PM.properties)
```

## Get physical properties dataframes

```python
df_coating_area, df_matrix_porosity = parse_strategies_to_df(strategy_dirs=strategy_dirs)
```

```python
summarize_dataframe(df_coating_area)
```

```python
df_matrix_porosity
```

## Get error dataframes

```python
df_coating_area_abs_error = get_absolute_error_df(df_coating_area, baseline_column="Masks")
df_matrix_porosity_abs_error = get_absolute_error_df(df_matrix_porosity, baseline_column="Masks")
```

```python
df_coating_area_rel_error = get_relative_error_df(df_coating_area, baseline_column="Masks")
df_matrix_porosity_rel_error = get_relative_error_df(df_matrix_porosity, baseline_column="Masks")
```

## Plot errors

```python
plot_model_errors(df_wide=df_coating_area_abs_error, 
                  output_dir=PM.statistics,
                  title="Absolute Error Distribution\nPolymer Coating Area", 
                  y_col="Absolute Error",
                  )
```

```python
plot_model_errors(df_wide=df_matrix_porosity_abs_error,
                  output_dir=PM.statistics,
                  title="Absolute Error Distribution\nMatrix Porosity",
                  y_col="Absolute Error",
                  )
```

```python
plot_model_errors(df_wide=df_coating_area_rel_error,
                  output_dir=PM.statistics,
                  title="Relative Error Distribution\nPolymer Coating Area", 
                  y_col="Relative Error (%)",
                  )
```

```python
plot_model_errors(df_wide=df_matrix_porosity_rel_error,
                  output_dir=PM.statistics,
                  title="Relative Error Distribution\nMatrix Porosity",
                  y_col="Relative Error (%)",
                  )
```

## Save dataframes

```python
save_dataframe_filename(df=df_coating_area.reset_index(), filename=RELATIVE_COATING_AREA_FILENAME, save_dir=PM.statistics)
save_dataframe_filename(df=df_coating_area_abs_error.reset_index(), filename=RELATIVE_COATING_AREA_FILENAME + "_abs_error", save_dir=PM.statistics)
save_dataframe_filename(df=df_coating_area_rel_error.reset_index(), filename=RELATIVE_COATING_AREA_FILENAME + "_rel_error", save_dir=PM.statistics)
```

```python
save_dataframe_filename(df=df_matrix_porosity.reset_index(), filename=RELATIVE_MATRIX_POROSITY_FILENAME, save_dir=PM.statistics)
save_dataframe_filename(df=df_matrix_porosity_abs_error.reset_index(), filename=RELATIVE_MATRIX_POROSITY_FILENAME + "_abs_error", save_dir=PM.statistics)
save_dataframe_filename(df=df_matrix_porosity_rel_error.reset_index(), filename=RELATIVE_MATRIX_POROSITY_FILENAME + "_rel_error", save_dir=PM.statistics)
```
