import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, measure
from scipy import ndimage as ndi
import czifile
from collections import defaultdict
from pathlib import Path

import json


# circularity filter
# or (4*np.pi*region.area)/region.perimeter^2 < drop_circ
# drop_circ=0.5,


def load_img(file, crop=None,
             display=False):  # defines function for image loading, takes file as argument as well as crop (to remove scale bar) and display option
    if file.suffix == ".czi":
        img = czifile.imread(file).squeeze()  # !!
        return img
        if len(img.shape) == 3:
            img = img[img.sum(axis=(1, 2)).argmax()]  # !!
    else:
        img = np.mean(img_as_float(io.imread(file)), axis=2)  # reads in .tifs

    if crop is not None:
        img = img[:crop]  # crops image if crop argument is given
    if display:
        plt.imshow(img, interpolation='none')
        plt.show()  # displays image for manual check
    return img


def get_droplet_mask(img, n_std, min_area=0, max_area=np.inf, display=False):
    im_mean = np.mean(img)
    im_std = np.std(img)
    std_from_mean = (img - im_mean) / im_std

    droplet_scaffold_mask = std_from_mean > int(n_std)
    droplet_mask = ndi.morphology.binary_fill_holes(droplet_scaffold_mask)
    droplet_mask_labeled = measure.label(droplet_mask)
    droplet_region_props = measure.regionprops(droplet_mask_labeled)
    for region in droplet_region_props:
        if region.area < min_area or region.area > max_area:
            droplet_mask[region.coords[:, 0], region.coords[:, 1]] = 0

    droplet_mask_labeled = measure.label(droplet_mask)
    droplet_region_props = measure.regionprops(droplet_mask_labeled, img)
    if display:
        plt.imshow(droplet_mask, interpolation='none')
        plt.show()
    return droplet_mask, droplet_region_props


def subtract_off_mean_background_intensity(img, nstd=0, min_area=0, max_area=np.inf, display=False):
    mask, props = get_droplet_mask(img, n_std=nstd, min_area=min_area, max_area=max_area)
    mean_background = img[~mask].mean()
    if display:
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(~mask)
        plt.show()
    return img - mean_background


def line_hist(data, label=None, **kwargs):
    hist, bin_edges = np.histogram(data, **kwargs)
    xaxis = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(xaxis, hist, marker="o", label=label)


def main(working_directory, display: bool = False, nstd: int = 2, min_area: int = 5, max_area: float = np.inf):
    working_directory = Path(working_directory)

    condensed_fractions = defaultdict(list)

    for concentration_dir in working_directory.iterdir():
        concentration = concentration_dir.name
        for file in concentration_dir.glob('*.tif'):
            img = load_img(file, display=display)
            droplet_mask, props = get_droplet_mask(img, nstd, min_area, max_area, False)
            print(f"proportion of image droplet: {np.mean(droplet_mask)}")
            condensed_fraction = np.mean(droplet_mask)
            condensed_fractions[concentration].append(condensed_fraction)

    total_integrated_intensity_stds = []
    concentrations, im_condensed_fractions = zip(*condensed_fractions.items())
    condensed_fraction_means = [np.mean(fractions) for fractions in im_condensed_fractions]
    condensed_fraction_stds = [np.std(fractions) for fractions in im_condensed_fractions]

    json.dump(condensed_fractions, open("output.json", "w"))

    plt.errorbar(concentrations, condensed_fraction_means, yerr=condensed_fraction_stds, linestyle='none', marker='o')
    plt.title(f"HsTFIIB-IDR at 10% PEG nstd= {nstd}, min area={min_area}, max area={max_area}")
    plt.xlabel("HsTFIIB-IDR M)")
    plt.ylabel("Condensed fraction")
    plt.ylim(0)
    plt.show()

    for concentration, fractions in condensed_fractions.items():
        concentrations = np.full(len(fractions), concentration)
        plt.scatter(concentrations, fractions, alpha=1)
    plt.title(f"mC-HsTFIIB at 20% PEG nstd= {nstd}, min area={min_area}, max area={max_area}")
    plt.xlabel("mC-HsTFIIB concentration (uM)")
    plt.ylabel("Condensed fraction")
    plt.ylim(0)
    plt.show()

    print()
