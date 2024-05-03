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
from typing_extensions import Annotated


def load_img(file_path: Path, convert_to_eight_bit: bool = True):
    image = ski.io.imread(str(file_path))

    # This handles when the microscope took two images
    if len(image.shape) > 3 and convert_to_eight_bit:
        image = image[0]

    # This handles when the microscope took two images but the images are 8 bit
    # np array shape gives (M, N, x) where x is length of colour array but doesn't exist in
    # 8 bit images
    elif len(image.shape) > 2 and not convert_to_eight_bit:
        image = image[0]

    if convert_to_eight_bit:
        image = ski.color.rgb2gray(image)

    return image


def threshold_via_std(image: np.array, n_std: int):
    """
    Creates a mask of the image where all pixels that are more than the specified number of standard deviations from the mean are set to true
    :param image:
    :param n_std:
    :return:
    """
    mean_intensity = np.mean(image)
    intensity_sd = np.std(image)

    # This is a normalisation step (you need to divide by the std)
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


def process_threshold(image: numpy.array, square_size: int = 2):
    """
    Given that not all pixels wihtin an otherwise homogenous region would be set to be apart of that region, we need to use a paint bucket to fill in those holes.
    :param image:
    :param square_size:
    :return:
    """
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
    logging.debug("Thresholding the image")
    if method == "STD":
        thresholded_img = threshold_via_std(image, n_std)

    elif method == "OTSU":
        thresholded_img = threshold_via_otsu(image)

    elif method == "MULTI_OTSU":
        thresholded_img = threshold_via_multiotsu(image)

    elif method == "ISODATA":
        thresholded_img = threshold_via_isodata(image)

    processed = process_threshold(thresholded_img, square_size)
    logging.debug("Thresholding complete")

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


def calc_mean_entropy(image):
    return np.mean(ski.filters.rank.entropy(image.astype("uint8"), ski.morphology.disk(3)))


def process_image(image: Path, output_dir: Path, method: str = "STD", square_size: int = 3, metadata: list = [],
                  debug=False, cv_thresh=0, convert_to_eight_bit: bool = False):
    img_name = image.stem
    image = load_img(image, convert_to_eight_bit=convert_to_eight_bit)

    logging.debug(f"Processing {img_name}")

    img_cv = np.std(image) / np.mean(image)
    regions = []

    # This step is doing nothing to the image. If we wanted to pre-process in some way
    # we could add that to the list of steps
    processed_img = preprocess_image(image, [])
    thresholded_image = threshold_image(processed_img, method, square_size=square_size)

    # TLDR: Don't use the code below. It allows for an iterative approach to the std
    # thresholding approaches

    # entropy_thresh = 0.07
    # multiplier = 2
    #
    # while calc_mean_entropy(thresholded_image) > entropy_thresh:
    #     logging.debug(f"Entropy too high, increasing threshold by {multiplier}")
    #     thresholded_image = processed_img > multiplier * np.std(processed_img)
    #     multiplier += 1

    logging.debug(f"Post thresholding complete")

    img_cf = get_condensed_fraction(thresholded_image)

    image_regions = get_image_regions(thresholded_image, image)

    mean_region_intensities = np.bincount(thresholded_image.astype(int).ravel(), processed_img.ravel()) / \
                              np.unique(thresholded_image.astype(int), return_counts=True)[1]

    # This accounts for occasions where the thresholding has failed to segment the image
    if len(np.unique(ski.measure.label(thresholded_image))) <= 0:
        logging.debug("Too few regions, setting CF to 0")
        img_cf = 0
        image_regions = []

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
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(thresholded_image)
        fig.savefig(intermediate_folder / f"{img_name}_thresholded.png")
        plt.close(fig)

    condensed_fraction = [img_cf, np.mean(image), np.std(image), np.max(image), np.min(image), calc_mean_entropy(image),
                          mean_region_intensities[0], mean_region_intensities[1]]
    condensed_fraction.extend(metadata)

    return condensed_fraction, regions


def process_dir(directory: Path, output_dir: Path, method: str = "STD", metadata: str = "", threads=1, debug=False,
                file_matching_pattern=[".tif"], n_std: int = 2, cv_thresh: int = 0, convert_to_eight_bit: bool = True):
    # Create the output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    metadata = metadata.split(",")

    regions = []
    condensed_fractions = []

    # Walk through different experimental conditions, each contained within its own directory
    for sub_directory in directory.glob("*"):
        logging.debug(f"Processing {sub_directory.stem}")
        images = []

        # Build up a list of images to process
        for pattern in file_matching_pattern:
            images.extend(sub_directory.glob(pattern))

        # Process each image
        for image in images:
            new_metadata = metadata.copy()
            # Make a note of the directory and image name to save to the metadata
            new_metadata.append(sub_directory.stem)
            new_metadata.append(image.stem)

            condensed_fraction, image_regions = process_image(image, output_dir, method, 3, new_metadata, debug=debug,
                                                              cv_thresh=cv_thresh,
                                                              convert_to_eight_bit=convert_to_eight_bit)

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
        "mean_entropy",
        "mean_bg_intensity",
        "mean_fg_intensity",
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
    labels = ski.measure.label(thresholded_image)
    image_overlay = ski.color.label2rgb(labels, image=image, alpha=0.5, bg_label=0, colors=[[91, 8, 136]])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_overlay)

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


def main(directory: str, output_dir: str, method: str = "STD", metadata: str = "", debug: bool = False,
         multichannel: bool = False, channels: str = "BLUE1,RED", n_std: int = 3, convert_to_eight_bit: bool = True):
    channel_lookup = {
        "BLUE1": "*Blue1.tif",
        "BLUE2": "*Blue2.tif",
        "BLUE3": "*Blue3.tif",
        "BLUE": "*Blue.tif",
        "BLUE6": "*Blue6.tif",
        "RED": "*Red.tif",
        "GREEN": "*Green.tif",
        "MERGED": "*RGB.tif"
    }

    file_extensions = ["*.tif"]

    if multichannel:
        file_extensions = []
        for channel in channels.split(","):
            file_extensions.append(channel_lookup[channel.strip()])

    process_dir(Path(directory), Path(output_dir), method=method, metadata=metadata, debug=debug,
                file_matching_pattern=file_extensions, convert_to_eight_bit=convert_to_eight_bit, n_std=n_std)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    typer.run(main)
