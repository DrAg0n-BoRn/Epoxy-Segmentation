from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Literal

from dragon.path_manager import list_files_by_extension_global, make_fullpath, sanitize_filename
from dragon.utilities import load_dataframe, merge_dataframes

from .format_model_name import _format_model_name


__all__ = [
    "get_metrics",
    "save_metrics_plot",
    "save_combined_metrics_plot"
]


CHOSEN_COLUMNS = ["Class", "Dice", "IoU"]

IGNORED_STRATEGIES = ["Tversky", "0_1-1_0", "0_5-0_5", "1_0-0_3"]

# COLUMN_RENAME_DICT = {
#     "pix_acc": "Pixel Accuracy",
#     "dice_weighted": "Dice Weighted",
#     "iou_weighted": "IoU Weighted",
#     "dice_macro": "Dice Macro",
#     "iou_macro": "IoU Macro"
# }

def get_metrics(models_dict: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns `test_df` and `validation_df` containing the metrics for all models in `models_dict`."""
    
    test_df = pd.DataFrame(columns=["Strategy"] + CHOSEN_COLUMNS)
    validation_df = test_df.copy(deep=True)
    
    column_order = test_df.columns.tolist()
    
    for raw_model_name, model_path in models_dict.items():
        
        if raw_model_name in IGNORED_STRATEGIES:
            continue

        model_name = _format_model_name(raw_model_name, wrap_length=100)
        
        all_csv_files = list_files_by_extension_global(directory=model_path, 
                                                       extension=".csv",
                                                       verbose=1)
        
        for csv_filename, csv_path in all_csv_files:
            if "per_class" in csv_filename:
                if "Test" in str(csv_path):            
                    df_raw, _ = load_dataframe(df_path=csv_path,
                                            kind="pandas",
                                            verbose=False)
                    df_raw["Strategy"] = model_name
                    
                    #reorder columns to match the order of `test_df`
                    df_raw = df_raw[column_order]
                    
                    test_df = merge_dataframes(test_df, df_raw, direction="vertical", verbose=False, reset_index=True)
                
                elif "Validation" in str(csv_path):
                    df_raw, _ = load_dataframe(df_path=csv_path,
                                            kind="pandas",
                                            verbose=False)
                    df_raw["Strategy"] = model_name
                    
                    #reorder columns to match the order of `validation_df`
                    df_raw = df_raw[column_order]
                    
                    validation_df = merge_dataframes(validation_df, df_raw, direction="vertical", verbose=False, reset_index=True)
    
    # reorder samples by strategy name alphabetically
    test_df = test_df.sort_values(by="Strategy").reset_index(drop=True)
    validation_df = validation_df.sort_values(by="Strategy").reset_index(drop=True)
    
    # rename columns for better readability
    # test_df = test_df.rename(columns=COLUMN_RENAME_DICT)
    # validation_df = validation_df.rename(columns=COLUMN_RENAME_DICT)
    
    return test_df, validation_df


def save_combined_metrics_plot(df_wide: pd.DataFrame,
                            save_dir: Union[str, Path],
                            filename: str) -> None:
    
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    sns.set_theme(style="whitegrid")
    
    # Set large font sizes
    plt.rcParams.update({
        'font.size': 14, 
        'axes.labelsize': 16, 
        'axes.titlesize': 18, 
        'xtick.labelsize': 14, 
        'ytick.labelsize': 14
    })
    
    df_melted = pd.melt(
        df_wide, 
        id_vars=["Strategy", "Class"], 
        value_vars=["Dice", "IoU"], 
        var_name="Metric", 
        value_name="Score"
    )
    
    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="Class",
        y="Score",
        hue="Strategy",
        col="Metric",
        palette="Set2",
        height=6,
        aspect=1.2
    )
    
    # g.set_axis_labels("Segmentation Class", "Score")
    g.set_axis_labels(x_var="", y_var="Score")
    g.set_titles("{col_name} Score per Class", weight="bold", pad=15)
    g.set(ylim=(0, 1.05), yticks=[i/10 for i in range(0, 11)])
    
    # 2. Safely apply explicit string formatting
    for ax in g.axes.flat:
        ax.set_yticklabels([f"{i/10:.1f}" for i in range(11)])
    
    g.tick_params(axis='x', rotation=45)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.01, 1), title="Strategy")
    
    sns.despine()
    
    plt.tight_layout()
    
    sanitized_filename = sanitize_filename(filename)
    if not sanitized_filename.endswith(".svg"):
        sanitized_filename += ".svg"
    
    g.savefig(save_path / sanitized_filename, format="svg", bbox_inches='tight')
    
    plt.close()


def save_metrics_plot(df: pd.DataFrame,
                    save_dir: Union[str, Path],
                    filename: str) -> None:
    
    save_path = make_fullpath(save_dir, make=True, enforce="directory")
    
    sns.set_theme(style="whitegrid")
    
    # Set large font sizes
    plt.rcParams.update({
        'font.size': 14, 
        'axes.labelsize': 16, 
        'axes.titlesize': 18, 
        'xtick.labelsize': 14, 
        'ytick.labelsize': 14
    })
    
    for metric in ["Dice", "IoU"]:
    
        fig, ax = plt.subplots(figsize=(12, 6))
            
        sns.barplot(
                data=df,
                x="Class",
                y=metric,
                hue="Strategy",
                palette="Set2",
                ax=ax
            )
        
        ax.set_title(f"{metric} Score per Class", weight="bold", pad=15)
        # ax.set_xlabel("Segmentation Class", labelpad=10)
        ax.set_xlabel("")
        ax.set_ylabel(metric, labelpad=10)
        
        ax.set_ylim(0, 1.05)
       # Set tick locations first, then apply formatted labels
        ax.set_yticks([i/10 for i in range(11)])
        ax.set_yticklabels([f"{i/10:.1f}" for i in range(11)])
        
        plt.legend(title="Strategy", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        
        sns.despine()
        
        sanitized_filename = sanitize_filename(filename)
        if not sanitized_filename.endswith(".svg"):
            sanitized_filename += ".svg"
        
        final_filename = f"{metric.lower()}_{sanitized_filename}"

        plt.savefig(save_path / final_filename, format="svg", bbox_inches='tight')
        
        plt.close(fig)
