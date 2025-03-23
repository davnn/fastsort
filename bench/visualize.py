import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from benchmark import BenchmarkResult
from matplotlib import cm


def plot_sorting_algorithms_time_comparison(data: BenchmarkResult, output_path: str) -> None:
    """Plots a comparison of sorting algorithms" mean execution times across various input types
    and saves the plot as a PNG file.

    Parameters
    ----------
    - data (dict[str, Any]): The input data containing sorting algorithms" execution times for different input types.
    - output_path (str): The path where the plot should be saved.

    """
    # Extract the relevant information from the data
    report = data["report"]
    names = [entry["name"] for entry in report]
    kinds = [entry["kind"] for entry in report]
    time_mean = [entry["time_mean"] for entry in report]
    time_std = [entry["time_std"] for entry in report]

    # Extract "fast" algorithm times for sorting
    fast_algorithm_times = {entry["name"]: entry["time_mean"] for entry in report if entry["kind"] == "fast"}

    # Sort unique_names based on the "fast" algorithm times
    unique_names = sorted(set(names), key=lambda name: fast_algorithm_times.get(name, float("inf")))

    # Create a unique list of sorting algorithms (e.g., quicksort, heapsort, etc.)
    unique_kinds = sorted(set(kinds))

    # Prepare the plot
    _, ax = plt.subplots(figsize=(16, 9))

    # Create a colormap
    cmap = cm.cividis  # type: ignore[reportAttributeAccessIssue]

    width = 0.215  # Width of the bars
    x_positions = np.arange(len(unique_names))  # X positions for input types

    # Plot bars for each algorithm kind (quicksort, heapsort, etc.)
    for idx, kind in enumerate(unique_kinds):
        kind_data = []
        kind_error = []

        for name in unique_names:
            # Collect the data corresponding to each input type and algorithm kind
            found = False
            for i in range(len(names)):
                if names[i] == name and kinds[i] == kind:
                    kind_data.append(time_mean[i])
                    kind_error.append(time_std[i])
                    found = True
                    break
            if not found:
                kind_data.append(np.nan)  # For missing combinations
                kind_error.append(np.nan)

        # Offset the x positions for each kind to avoid overlap
        bars = ax.bar(x_positions + width * idx, kind_data, width, label=kind, yerr=kind_error, capsize=5)

        # Set the colors of the bars using the colormap
        for i, bar in enumerate(bars):
            bar.set_color(cmap(idx / len(unique_kinds)))  # Normalize index to the colormap

        # Annotate each bar with percentage deviation from the "fast" algorithm
        for i, bar in enumerate(bars):
            if kind != "fast":  # Skip annotation for "fast" bars
                fast_time = fast_algorithm_times[unique_names[i]]
                pct_deviation = ((kind_data[i] - fast_time) / fast_time) * 100  # Percentage deviation

                # Get the top of the whisker (mean + std)
                whisker_top = kind_data[i] + kind_error[i]

                # Get the y-axis limits to check if annotation would be out of bounds
                y_min, y_max = ax.get_ylim()
                approx_text_height = (y_max - y_min) / 5

                if whisker_top + approx_text_height < y_max:
                    # Annotate the bar with the percentage deviation
                    ax.annotate(
                        f"{pct_deviation:+.1f}%",
                        (bar.get_x() + bar.get_width() / 2, whisker_top),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=10,
                        color="black",
                    )

    # Adding labels, title, and customizing the plot
    ax.set_ylabel("Mean Execution Time (seconds)", fontsize=16)
    sort_name = "argsort" if data["argsort"] else "sort"
    data_size = (data["size"] ** 2,) if data["flatten"] else (data["size"], data["size"])
    contiguous = "contiguous" if data["contiguous"] or data["flatten"] else "non-contiguous"
    ax.set_title(f"Comparison of {sort_name} algorithm with {contiguous} array of size {data_size}.", fontsize=14)
    ax.set_xticks(x_positions + width * (len(unique_kinds) - 1) / 2)  # Center the x-ticks
    ax.set_xticklabels(unique_names, rotation=20, ha="right", fontsize=12)

    # Displaying the legend
    ax.legend(title="Algorithm", loc="upper right", fontsize=12)

    # Add grid for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Save the plot as PNG
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_data_from_json(file_path: str) -> BenchmarkResult:
    """Loads the JSON data from the specified file path.

    Parameters
    ----------
    - file_path (str): The path to the JSON file to be loaded.

    Returns
    -------
    - dict: The data loaded from the JSON file.

    """
    with open(file_path) as file:
        data = json.load(file)
    return data


def process_reports(input_folder: str) -> None:
    """Iterates over all JSON files in the given folder, processes each report, and saves the plot.

    Parameters
    ----------
    - input_folder (str): Path to the folder containing the JSON report files.

    """
    # Iterate through all files in the folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        # Only process JSON files
        if file_name.endswith(".json"):
            try:
                # Load the JSON data
                data: BenchmarkResult = load_data_from_json(file_path)

                # Define output PNG file path
                output_file = os.path.splitext(file_name)[0] + ".png"
                output_path = os.path.join(input_folder, output_file)

                # Plot and save the comparison chart
                plot_sorting_algorithms_time_comparison(data, output_path)
                print(f"Saved plot for {file_name} as {output_file}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize sorting algorithm performance from JSON reports.")
    parser.add_argument("--folder", type=str, default="reports", help="Path to the folder containing JSON reports")

    # Parse arguments
    args = parser.parse_args()

    # Retrieve the path to the folder
    script_dir = Path(__file__).resolve().parent / args.folder

    # Process reports in the given folder
    process_reports(str(script_dir))
