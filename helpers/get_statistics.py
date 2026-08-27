import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Union
from pathlib import Path

from dragon.path_manager import make_fullpath, sanitize_filename, list_files_by_extension

from .constants import RELATIVE_COATING_AREA_FILENAME, RELATIVE_MATRIX_POROSITY_FILENAME
from .format_model_name import _format_model_name


__all__ = [
    "plot_model_errors",
    "convert_wide_to_long",
    "parse_strategies_to_df",
    "get_relative_error_df",
    "get_absolute_error_df"
]


def plot_model_errors(df_wide: pd.DataFrame, 
                      output_dir: Union[str, Path],
                      x_col="Semantic Segmentation Strategy", 
                      y_col="Error", 
                      title="Error Distribution", 
                      palette="Set2"):
    
    df_long = convert_wide_to_long(df=df_wide, 
                                   value_name=y_col, 
                                   variable_name=x_col)
    
    # Set large font sizes
    plt.rcParams.update({
        'font.size': 14, 
        'axes.labelsize': 16, 
        'axes.titlesize': 18, 
        'xtick.labelsize': 14, 
        'ytick.labelsize': 14
    })
    
    plt.figure(figsize=(10, 6))
    
    # Added palette and hue for individualized boxplot colors
    sns.boxplot(data=df_long, x=x_col, y=y_col, palette=palette, hue=x_col, legend=False, showfliers=False)
    
    sns.swarmplot(data=df_long, x=x_col, y=y_col, color="black", alpha=0.7, size=4)
    
    plt.title(title)
    plt.ylabel(y_col)
    plt.xlabel(x_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # despine the plot
    sns.despine()
    
    filename = sanitize_filename(title) + ".svg"
    output_dir = make_fullpath(output_dir, make=True, enforce="directory")
    
    save_path = output_dir / filename
    plt.savefig(save_path, format="svg", bbox_inches='tight')
    
    plt.close()


def convert_wide_to_long(df: pd.DataFrame, 
                         value_name="Error", 
                         variable_name="Semantic Segmentation Strategy",
                         index_name="Sample",
                         drop_unused_columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Converts a wide-format DataFrame to a long-format DataFrame suitable for plotting.
    
    Drop all unused columns before using this function.
    
    Args:
        df (pd.DataFrame): The wide-format DataFrame to convert.
        value_name (str): The name to assign to the value column in the long-format DataFrame.
        variable_name (str): The name to assign to the variable column in the long-format DataFrame.
        index_name (str): The name to assign to the index column in the long-format DataFrame.
        drop_unused_columns (list[str], optional): List of column names to drop from the DataFrame before conversion. If None, no columns will be dropped.
    
    Returns:
        pd.DataFrame: The long-format DataFrame.
    """
    if drop_unused_columns:
        df = df.drop(columns=drop_unused_columns, errors='ignore')
        
    df_reset = df.rename_axis(index=index_name).reset_index()
    
    df_long = pd.melt(
        df_reset, 
        id_vars=[index_name], 
        var_name=variable_name, 
        value_name=value_name
    )
    
    return df_long

def _load_json_to_df(filepath, column_name):
    # typ='series' forces pandas to treat the keys as the index
    # to_frame() converts it to a DataFrame and names the single column
    df = pd.read_json(filepath, typ='series').to_frame(name=column_name)
    return df


def parse_strategies_to_df(strategy_dirs: dict[str, Path],):
    """
    Parses the JSON files in the provided strategy directories and compiles them into two DataFrames:
    
    1. Coating Area DataFrame: Contains the relative coating area for each strategy.
    2. Matrix Porosity DataFrame: Contains the relative matrix porosity for each strategy.
    
    Args:
        strategy_dirs (dict): A dictionary where keys are strategy names and values are the corresponding directory paths containing the JSON files.
    
    Returns:
        tuple: A tuple containing two DataFrames:
            - coating_area_data: DataFrame with relative coating area for each strategy.
            - matrix_porosity_data: DataFrame with relative matrix porosity for each strategy.
    """
    # build dataframes
    coating_area_data: pd.DataFrame = pd.DataFrame()
    matrix_porosity_data: pd.DataFrame = pd.DataFrame()

    for strategy, strategy_subdir in strategy_dirs.items():
        
        # format strategy name for better readability
        strategy = _format_model_name(strategy)
        
        # find json files
        all_json_files = list_files_by_extension(directory=strategy_subdir, 
                                                extension="json",
                                                verbose=False)
        
        # save data to appropriate dictionaries
        for json_filename, json_path in all_json_files.items():
            if json_filename == RELATIVE_COATING_AREA_FILENAME:
                df = _load_json_to_df(filepath=json_path, column_name=strategy)
                coating_area_data = pd.concat([coating_area_data, df], axis=1)

            elif json_filename == RELATIVE_MATRIX_POROSITY_FILENAME:
                df = _load_json_to_df(filepath=json_path, column_name=strategy)
                matrix_porosity_data = pd.concat([matrix_porosity_data, df], axis=1)
    
    # reorder columns alphabetically
    coating_area_data = coating_area_data.reindex(sorted(coating_area_data.columns), axis=1)
    matrix_porosity_data = matrix_porosity_data.reindex(sorted(matrix_porosity_data.columns), axis=1)
    
    # sort by index (image names)
    coating_area_data = coating_area_data.sort_index()
    matrix_porosity_data = matrix_porosity_data.sort_index()
    
    return coating_area_data, matrix_porosity_data


def get_relative_error_df(df: pd.DataFrame, baseline_column: str = "Masks") -> pd.DataFrame:
    baseline = df[baseline_column]
    df_metrics = df.drop(columns=[baseline_column])
    
    # Vectorized relative error calculation
    return 100 * df_metrics.sub(baseline, axis=0).abs().div(baseline, axis=0)


def get_absolute_error_df(df: pd.DataFrame, baseline_column: str = "Masks") -> pd.DataFrame:
    baseline = df[baseline_column]
    df_metrics = df.drop(columns=[baseline_column])
    
    # Vectorized absolute error calculation
    return df_metrics.sub(baseline, axis=0).abs()
