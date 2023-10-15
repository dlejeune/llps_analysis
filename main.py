import numpy
import pandas as pd
import numpy as np
import skimage as ski
from pathlib import Path
import typer
import matplotlib.pyplot as plt


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


def process_threshold(image: numpy.array):
    filled_holes = ski.morphology.closing(image, ski.morphology.square(3))

    return filled_holes


def process_image(image_path: Path, method: str = "STD"):
    image = load_img(image_path)

    if method == "STD":
        thresholded_img = threshold_via_std(image, 2)

    elif method == "OTSU":
        thresholded_img = threshold_via_otsu(image)

    processed = process_threshold(thresholded_img)

    labelled_image = ski.measure.label(processed)
    region_props = ski.measure.regionprops(labelled_image, processed)

    mean_area = np.mean([region.area for region in region_props])
    prop_area = np.sum([region.area for region in region_props]) / processed.size

    print(f"Mean {np.sum(processed == 1) / processed.size}. Average Area: {mean_area}. Prop Area: {prop_area}")

    return processed


def display(image: np.array):
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def get_condensed_fraction(image: np.array):
    return np.sum(image == 1) / image.size


def main():
    image = Path("data/small_condensates.tif")
    std_processed = process_image(image, "STD")
    otsu_processed = process_image(image, "OTSU")
    print(f"STD: {get_condensed_fraction(std_processed)}\n OTSU: {get_condensed_fraction(otsu_processed)}")

    display(std_processed)
    display(otsu_processed)


if __name__ == "__main__":
    typer.run(main)
