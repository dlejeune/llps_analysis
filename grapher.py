from pathlib import Path
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as pn


def fix_concentration(row, stock_concentrations):
    return stock_concentrations[row["prep"]] / row["dir_name"]


def plot_mean_std_err(data, x, y, x_axis, y_axis, legend, output_dir, graph_type):
    modified_data = data.copy()
    modified_data = modified_data.groupby(["prep", "conc"])[x].aggregate(["mean", "std", "count"]).reset_index()
    modified_data["x_std_err"] = modified_data["std"] / np.sqrt(modified_data["count"])
    modified_data["mean_plus_std_err"] = modified_data["mean"] + modified_data["x_std_err"]
    modified_data["mean_minus_std_err"] = modified_data["mean"] - modified_data["x_std_err"]
    tbp_region_area_plot = (
            pn.ggplot(modified_data,
                      pn.aes(x="conc", y="mean", color="prep")) +
            pn.geom_point() +
            pn.geom_errorbar(pn.aes(ymin="mean_minus_std_err", ymax="mean_plus_std_err")) +
            pn.labs(x="Concentration ($\mu$M)", y="Area (pixels)") +
            pn.theme(figure_size=(8, 4.5))
        # pn.scale_x_log10()
    )
    tbp_region_area_plot


def main(directory: Path, output_dir: Path, graph_type: str, concentrations: dict, y_axis: str, x_axis: str,
         legend: str, y_exclude: int = np.inf, x_exclude: int = np.inf):
    """Graphs the data from the csv files in the directory"""

    regions_dfs = []
    condense_fraction_dfs = []

    for key in concentrations.keys():
        regions_dfs.append(pd.read_csv(concentrations / key / f"{key}_regions.csv"))
        condense_fraction_dfs.append(pd.read_csv(concentrations / key / f"{key}_condensed_fractions.csv"))

    regions = pd.concat(regions_dfs)
    condensed_fractions = pd.concat(condense_fraction_dfs)

    condensed_fractions["conc"] = condensed_fractions.apply(lambda x: fix_concentration(x, concentrations), axis=1)
    regions["conc"] = regions.apply(lambda x: fix_concentration(x, concentrations), axis=1)

    regions["circularity"] = (np.square(regions["perimeter"])) / (
            4 * np.pi * regions["area"])
