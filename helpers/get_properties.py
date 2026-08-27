import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from pathlib import Path

from dragon.path_manager import make_fullpath
from dragon.IO_tools import save_json

from .constants import CLASS_FIBER_EPOXY_MATRIX, CLASS_POLYMER_COATING, CLASS_VOIDS, RELATIVE_MATRIX_POROSITY_FILENAME, RELATIVE_COATING_AREA_FILENAME


__all__ = [
    "save_relative_properties"
]


## helper visualization function to plot properties
def _plot_properties(data: dict[str, Union[int, float]], 
                    ylabel: str, 
                    filename: str,
                    output_dir: Path,
                    show_plot: bool = False):
    """
    Generates a bar plot for the provided properties and saves it to disk.
    
    Args:
        data (dict): A dictionary where keys are image names and values are their corresponding values.
        ylabel (str): The label for the y-axis of the plot.
        filename (str): The name of the file to save the plot as.
        output_dir (Path): The directory where the plot will be saved.
        show_plot (bool): Whether to display the plot after saving.
    """
    with plt.rc_context({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14}):
        plt.figure(figsize=(12, 6))
        
        # Sort the dictionary by values in descending order
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
        keys = list(sorted_data.keys())
        values = list(sorted_data.values())
        sns.barplot(x=keys, y=values, hue=keys, palette="husl", legend=False)
        
        plt.title('')
        plt.xlabel('')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        
        sns.despine()
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
        if show_plot:
            plt.show()
        plt.close()


def _get_relative_matrix_porosity(all_images: dict[str, dict[str, int]]) -> dict[str, float]:
    """
    Calculates the relative porosity of the core matrix for each image.
    
    The relative porosity is calculated as the ratio of the number of void pixels to the total number of pixels in the core matrix.
    
    Args:
        all_images (dict): A dictionary where keys are image names and values are dictionaries containing pixel counts for different classes.

    Returns:
        dict: A dictionary where keys are image names and values are the calculated matrix porosities as relative ratios.
    """
    core_matrix_porosity: dict[str, float] = {}

    for img_name, pixel_counts in all_images.items():
        current_porosity = pixel_counts[CLASS_VOIDS] / (pixel_counts[CLASS_FIBER_EPOXY_MATRIX] + pixel_counts[CLASS_VOIDS])
        core_matrix_porosity[img_name] = current_porosity
    
    return core_matrix_porosity


def _get_relative_polymer_coating_area(all_images: dict[str, dict[str, int]]) -> dict[str, float]:
    """
    Calculates the relative area of the polymer coating for each image.

    The relative area is calculated as the ratio of the number of polymer coating pixels to the total number of pixels in the core matrix.

    Args:
        all_images (dict): A dictionary where keys are image names and values are dictionaries containing pixel counts for different classes.

    Returns:
        dict: A dictionary where keys are image names and values are the calculated polymer coating areas as relative ratios.
    """
    relative_coating_area: dict[str, float] = {}

    for img_name, pixel_counts in all_images.items():
        current_area = pixel_counts[CLASS_POLYMER_COATING] / (pixel_counts[CLASS_FIBER_EPOXY_MATRIX] + pixel_counts[CLASS_VOIDS] + pixel_counts[CLASS_POLYMER_COATING])
        relative_coating_area[img_name] = current_area
    
    return relative_coating_area


def save_relative_properties(images_pixel_counts: dict[str, dict[str, int]], output_dir: Union[str, Path]):
    """
    Calculates and saves the relative properties of the images to disk.

    This function calculates the relative matrix porosity and relative polymer coating area for each image,
    saves the results as JSON files, and generates bar plots for visualization.

    Args:
        all_images (dict): A dictionary where keys are image names and values are dictionaries containing pixel counts for different classes.
        output_dir (Path): The directory where the results will be saved.
    """
    output_path = make_fullpath(output_dir, make=True, enforce="directory")
    
    # Calculate relative properties
    core_matrix_porosity = _get_relative_matrix_porosity(images_pixel_counts)
    relative_coating_area = _get_relative_polymer_coating_area(images_pixel_counts)

    # Save results as JSON
    save_json(data=core_matrix_porosity, 
          directory=output_path, 
          filename=RELATIVE_MATRIX_POROSITY_FILENAME + ".json")
    
    save_json(data=relative_coating_area, 
          directory=output_path, 
          filename=RELATIVE_COATING_AREA_FILENAME + ".json")

    # Generate and save plots
    _plot_properties(core_matrix_porosity, ylabel="Relative Matrix Porosity", filename=RELATIVE_MATRIX_POROSITY_FILENAME + ".svg", output_dir=output_path, show_plot=False)
    _plot_properties(relative_coating_area, ylabel="Relative Polymer Coating Area", filename=RELATIVE_COATING_AREA_FILENAME + ".svg", output_dir=output_path, show_plot=False)
