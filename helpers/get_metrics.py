from pathlib import Path
import pandas as pd

from dragon.path_manager import list_files_by_extension_global
from dragon.utilities import load_dataframe, merge_dataframes

from .format_model_name import _format_model_name


CHOSEN_COLUMNS = ["pix_acc", "dice_weighted", "iou_weighted", "dice_macro","iou_macro"]

COLUMN_RENAME_DICT = {
    "pix_acc": "Pixel Accuracy",
    "dice_weighted": "Dice Weighted",
    "iou_weighted": "IoU Weighted",
    "dice_macro": "Dice Macro",
    "iou_macro": "IoU Macro"
}

def get_metrics(models_dict: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns `test_df` and `validation_df` containing the metrics for all models in `models_dict`."""
    
    test_df = pd.DataFrame(columns=["Strategy"] + CHOSEN_COLUMNS)
    validation_df = test_df.copy(deep=True)
    
    column_order = test_df.columns.tolist()
    
    for raw_model_name, model_path in models_dict.items():
        model_name = _format_model_name(raw_model_name, wrap_length=100)
        
        all_csv_files = list_files_by_extension_global(directory=model_path, 
                                                       extension=".csv",
                                                       verbose=1)
        
        for csv_filename, csv_path in all_csv_files:
            if "global" in csv_filename:
                if "Test" in str(csv_path):            
                    df_raw, _ = load_dataframe(df_path=csv_path,
                                            use_columns=CHOSEN_COLUMNS,
                                            kind="pandas",
                                            verbose=False)
                    df_raw["Strategy"] = model_name
                    
                    #reorder columns to match the order of `test_df`
                    df_raw = df_raw[column_order]
                    
                    test_df = merge_dataframes(test_df, df_raw, direction="vertical", verbose=False, reset_index=True)
                
                elif "Validation" in str(csv_path):
                    df_raw, _ = load_dataframe(df_path=csv_path,
                                            use_columns=CHOSEN_COLUMNS,
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
    test_df = test_df.rename(columns=COLUMN_RENAME_DICT)
    validation_df = validation_df.rename(columns=COLUMN_RENAME_DICT)
    
    return test_df, validation_df
