import numpy
import pandas as pd
import numpy as np
import skimage as ski
from pathlib import Path
import typer
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from rich.progress import track
import logging


def load_img(file_path: Path, gray: bool = True):
    image = ski.io.imread(str(file_path))

    # This handles when the microscope took two images
    if len(image.shape) > 3:
        image = image[0]

    if gray:
        image = ski.color.rgb2gray(image)

    return image


def threshold_via_std(image: np.array, n_std: int):
    mean_intensity = np.mean(image)
    intensity_sd = np.std(image)

    img_dist_from_mean = (image - mean_intensity) / intensity_sd

    droplet_boolean_mask = img_dist_from_mean > n_std

    droplet_integer_mask = droplet_boolean_mask.astype(int)

    return droplet_integer_mask


def threshold_via_isodata(image: np.array):
    droplet_boolean_mask = image > ski.filters.threshold_isodata(image)

    droplet_integer_mask = droplet_boolean_mask.astype(int)

    return droplet_integer_mask


def threshold_via_otsu(image: np.array):
    droplet_boolean_mask = image > ski.filters.threshold_otsu(image)

    droplet_integer_mask = droplet_boolean_mask.astype(int)

    return droplet_integer_mask


def threshold_via_multiotsu(image: np.array):
    thresholds = ski.filters.threshold_multiotsu(image)
    regions = np.digitize(image, bins=thresholds)

    img = regions == np.max(regions)

    return img


def threshold_via_niblack(image: np.array, window_size: int, k: int):
    thresh_niblack = ski.filters.threshold_niblack(image, window_size=window_size, k=k)
    binary_img_niblack = image > thresh_niblack

    droplet_integer_mask = binary_img_niblack.astype(int)

    return droplet_integer_mask


def threshold_via_chan_vese(image: np.array, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200, dt=0.5,
                            init_level_set="checkerboard", extended_output=True):
    cv = ski.segmentation.chan_vese(image, mu=0.25, lambda1=lambda1, lambda2=lambda2, tol=tol,
                                    max_num_iter=max_num_iter,
                                    dt=dt, init_level_set=init_level_set, extended_output=extended_output)
    return cv[0]


def process_threshold(image: numpy.array, square_size: int = 3):
    filled_holes = ski.morphology.closing(image, ski.morphology.square(square_size))

    return filled_holes


def preprocess_image(image: np.array, steps: list):
    logging.debug("Preprocessing the image")
    image_start = image.copy()

    if "median" in steps:
        image_start = ski.filters.median(image_start, np.ones((3, 3)))

    if "richardson" in steps:
        image_start = ski.restoration.richardson_lucy(image_start, np.ones((3, 3)))

    if "log_contrast" in steps:
        image_start = ski.exposure.adjust_log(image_start, 1)

    if "adapt_hist" in steps:
        image_start = ski.exposure.equalize_adapthist(image_start)

    if "closing" in steps:
        image_start = ski.morphology.closing(image_start, ski.morphology.square(3))

    logging.debug("Preprocessing complete")
    return image_start


def threshold_image(image: np.array, method: str = "STD", n_std: int = 2, square_size: int = 3):
    if method == "STD":
        thresholded_img = threshold_via_std(image, n_std)

    elif method == "OTSU":
        thresholded_img = threshold_via_otsu(image)

    elif method == "MULTI_OTSU":
        thresholded_img = threshold_via_multiotsu(image)

    elif method == "ISODATA":
        thresholded_img = threshold_via_isodata(image)

    processed = process_threshold(thresholded_img, square_size)

    return processed


def get_image_regions(image: np.array, original_image: np.array = None):
    cleared_image = ski.segmentation.clear_border(image)
    labelled_image = ski.measure.label(cleared_image)

    if original_image is not None:
        region_props = ski.measure.regionprops(labelled_image, original_image)
    else:
        region_props = ski.measure.regionprops(labelled_image, image)

    return region_props


def display(image: np.array):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def get_condensed_fraction(image: np.array):
    return np.sum(image == 1) / image.size


def process_image(image: Path, output_dir: Path, method: str = "STD", square_size: int = 3, metadata: list = [],
                  debug=False):
    img_name = image.stem
    image = load_img(image)

    logging.debug(f"Processing {img_name}")

    img_cv = np.std(image) / np.mean(image)
    regions = []

    # Hardcoded, naughty
    logging.debug(f"Image CV: {img_cv}")
    if img_cv < 0.2:
        logging.debug("Coeff of Variance < 0.1, skipping")
        img_cf = 0
        image_regions = []

    else:

        processed_img = preprocess_image(image, ["adapt_hist", "richardson" "closing"])
        thresholded_image = threshold_image(processed_img, method, square_size=square_size)
        img_cf = get_condensed_fraction(thresholded_image)
        image_regions = get_image_regions(thresholded_image, image)

        for prop in image_regions:
            temp = []
            temp.append(prop.area)
            temp.append(prop.mean_intensity)
            temp.append(prop.perimeter)
            temp.append(prop.axis_major_length)
            temp.append(prop.axis_minor_length)
            temp.extend(metadata)
            regions.append(temp)

        if debug:
            intermediate_folder = output_dir / "intermediate"
            intermediate_folder.mkdir(exist_ok=True, parents=True)
            fig, region_image = draw_regions_on_image(image, thresholded_image, image_regions)
            fig.savefig(intermediate_folder / f"{img_name}_regions.png")

    condensed_fraction = [img_cf, np.mean(image), np.std(image), np.max(image), np.min(image)]
    condensed_fraction.extend(metadata)

    return condensed_fraction, regions


def process_dir(directory: Path, output_dir: Path, method: str = "STD", metadata: str = "", threads=1, debug=False):
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    metadata = metadata.split(",")

    regions = []
    condensed_fractions = []

    for sub_directory in directory.glob("*"):
        logging.debug(f"Processing {sub_directory.stem}")
        for image in sub_directory.glob("*.tif"):
            new_metadata = metadata.copy()
            new_metadata.append(sub_directory.stem)
            new_metadata.append(image.stem)
            condensed_fraction, image_regions = process_image(image, output_dir, method, 3, new_metadata, debug=debug)

            regions.extend(image_regions)
            condensed_fractions.append(condensed_fraction)

    region_columns = ["area",
                      "mean_intensity",
                      "perimeter",
                      "axis_major_length",
                      "axis_minor_length",
                      "prep",
                      "dir_name",
                      "image_name"]

    cf_columns = [
        "condensed_fraction",
        "mean_intensity",
        "std_intensity",
        "max_intensity",
        "min_intensity",
        "prep",
        "dir_name",
        "image_name"
    ]

    region_df = pd.DataFrame(regions, columns=region_columns)
    condensed_fraction_df = pd.DataFrame(condensed_fractions, columns=cf_columns)

    region_df.to_csv(output_dir / f"{metadata[0]}_regions.csv", index=False)
    condensed_fraction_df.to_csv(output_dir / f"{metadata[0]}_condensed_fractions.csv", index=False)

    return condensed_fractions, regions


def compute_and_draw_regions_on_image(image, threshold):
    thresholded_image = image > threshold
    regions = get_image_regions(thresholded_image, image)

    return draw_regions_on_image(image, thresholded_image, regions)


def draw_regions_on_image(image, thresholded_image, regions):
    image_overlay = ski.color.label2rgb(thresholded_image, image, alpha=0.5, bg_label=0, bg_color=None,
                                        colors=[(1, 0, 0)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for region in regions:
        # take regions with large enough areas
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    return fig, ax


def compute_over_multiple_dirs(parent_dir: str, parent_output_dir: str = "data", method: str = "OTSU",
                               metadata: str = "", debug: bool = False):
    for experiment_dir in Path(parent_dir).glob("*"):
        if experiment_dir.is_dir():
            experiment_output_dir = Path(parent_output_dir) / experiment_dir.stem
            experiment_output_dir.mkdir(exist_ok=True, parents=True)

            process_dir(experiment_dir, experiment_output_dir, method=method, metadata=experiment_dir.stem, debug=debug)


def main(mode, directory: str, output_dir: str, method: str = "OTSU", metadata: str = "", debug: bool = False):
    if mode == "dir":
        process_dir(Path(directory), Path(output_dir), method=method, metadata=metadata, debug=debug)
    elif mode == "multi":
        compute_over_multiple_dirs(directory, output_dir, method=method, metadata=metadata, debug=debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    typer.run(main)
