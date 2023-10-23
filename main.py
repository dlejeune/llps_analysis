import numpy
import pandas as pd
import numpy as np
import skimage as ski
from pathlib import Path
import typer
import matplotlib.pyplot as plt
from rich.progress import track


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


def threshold_image(image: np.array, method: str = "STD", n_std: int = 2, square_size: int = 3):
    if method == "STD":
        thresholded_img = threshold_via_std(image, n_std)

    elif method == "OTSU":
        thresholded_img = threshold_via_otsu(image)

    elif method == "MULTI_OTSU":
        thresholded_img = threshold_via_multiotsu(image)

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


def process_image(image: Path, method: str = "STD", square_size: int = 3, metadata: list = []):
    image = load_img(image)
    thresholded_image = threshold_image(image, method, square_size=square_size)

    condensed_fraction = []
    condensed_fraction.append(get_condensed_fraction(thresholded_image))
    condensed_fraction.extend(metadata)

    regions = []
    image_regions = get_image_regions(thresholded_image, image)

    image_overlay = ski.color.label2rgb(thresholded_image, image, alpha=0.5, bg_label=0, bg_color=None,
                                        colors=[(1, 0, 0)])

    for prop in image_regions:
        temp = []
        temp.append(prop.area)
        temp.append(prop.mean_intensity)
        temp.append(prop.perimeter)
        temp.append(prop.axis_major_length)
        temp.append(prop.axis_minor_length)
        temp.extend(metadata)
        regions.append(temp)

    return condensed_fraction, regions


def process_dir(directory: Path, method: str = "STD", square_size: int = 3, metadata: list = []):
    regions = []
    condensed_fractions = []

    for sub_directory in directory.glob("*"):
        for image in track(sub_directory.glob("*.tif")):
            new_metadata = metadata.copy()
            new_metadata.append(sub_directory.stem)
            new_metadata.append(image.stem)
            condensed_fraction, image_regions = process_image(image, method, square_size, new_metadata)

            regions.extend(image_regions)
            condensed_fractions.append(condensed_fraction)

    return condensed_fractions, regions


def main(directory: str, output_dir: str = "data", method: str = "OTSU", metadata: str = ""):
    metadata = metadata.split(",")
    image = Path(directory)
    condensed_fractions, regions = process_dir(image, method=method, metadata=metadata)

    region_columns = ["area",
                      "mean_intensity",
                      "perimeter",
                      "axis_major_length",
                      "axis_minor_length",
                      "prep",
                      "dir_name",
                      "image_name"]

    region_df = pd.DataFrame(regions,
                             columns=region_columns)
    condensed_fraction_df = pd.DataFrame(condensed_fractions,
                                         columns=["condensed_fraction", "prep", "dir_name", "image_name"])

    output_dir = Path(output_dir)
    region_df.to_csv(output_dir / "regions.csv", index=False)
    condensed_fraction_df.to_csv(output_dir / "condensed_fractions.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
