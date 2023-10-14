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


def process_threshold(image: numpy.array):
    filled_holes = ski.morphology.closing(image, ski.morphology.square(3))

    return filled_holes


def process_image(image_path: Path, method: str = "STD"):
    image = load_img(image_path)
    thresholded_img = threshold_via_std(image, 2)

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


def main():
    image = Path("data/good_condensates.tif")
    processed = process_image(image)
    display(processed)


if __name__ == "__main__":
    typer.run(main)
