"""Script to generate SOD kernel from Meentemeyer."""

import os
from collections import Counter
import numpy as np
from scipy import optimize
import raster_tools


def meentemeyer_kernel(dist, params):
    """Calculate kernel used in Meentemeyer 2011."""

    gamma = params['gamma']
    alpha_1 = params['alpha_1']
    alpha_2 = params['alpha_2']

    norm_factor_1 = 2/(np.pi*alpha_1)
    norm_factor_2 = 2/(np.pi*alpha_2)

    kernel_value = (gamma * norm_factor_1 * (1 / (1 + (dist / alpha_1)**2)) +
                    (1 - gamma) * norm_factor_2 * (1 / (1 + (dist / alpha_2)**2)))

    return kernel_value


def meentemeyer_cdf(dist, params):
    """Calculate CDF of Meentemeyer kernel."""

    gamma = params['gamma']
    alpha_1 = params['alpha_1']
    alpha_2 = params['alpha_2']

    norm_factor_1 = 2/(np.pi*alpha_1)
    norm_factor_2 = 2/(np.pi*alpha_2)

    cdf_value = (gamma * norm_factor_1 * alpha_1 * np.arctan(dist / alpha_1) +
                 (1 - gamma) * norm_factor_2 * alpha_2 * np.arctan(dist / alpha_2))

    return cdf_value


def kernel_sample(params, num=1):
    """Generate sample(s) from Meentemeyer kernel."""

    uniform_sample = np.random.rand(num)

    samples = np.array([optimize.root(lambda x: (meentemeyer_cdf(x, params) - unif_sam), x0=0).x[0]
                        for unif_sam in uniform_sample])

    return samples


def get_kernel_array(dist_samples, angle_samples, kernel_range=10, resolution=250):
    """Extract array kernel from kernel samples"""

    n_samples = len(dist_samples)

    sample_x = (resolution/2) + (dist_samples * np.cos(angle_samples))
    sample_y = (resolution/2) + (dist_samples * np.sin(angle_samples))

    cell_x = [int(x // resolution) + kernel_range for x in sample_x]
    cell_y = [int(y // resolution) + kernel_range for y in sample_y]

    all_cells = zip(cell_x, cell_y)
    counter_dict = Counter(all_cells)

    kernel_array = np.array([[counter_dict[(x, y)] / n_samples for y in range(2*kernel_range+1)]
                             for x in range(2*kernel_range+1)])

    return kernel_array


def generate_kernel():
    """Generate ASCII raster format kernel for SOD simulations."""

    meentemeyer_params = {
        'gamma': 0.9947,
        'alpha_1': 20.57,
        'alpha_2': 9500
    }

    num_samples = int(1e8)
    host_raster_eroi = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "EROI_Landscape", "HostDensity.txt"
    ))
    kernel_range = max(host_raster_eroi.array.shape)

    all_dist_samples = kernel_sample(meentemeyer_params, num=num_samples)
    all_angle_samples = np.random.rand(num_samples) * 2 * np.pi

    kernel_array = get_kernel_array(all_dist_samples, all_angle_samples, kernel_range=kernel_range)
    kernel_raster = raster_tools.RasterData(
        (2*kernel_range+1, 2*kernel_range+1), array=kernel_array, cellsize=250)
    kernel_raster.to_file(os.path.join("GeneratedData", "Kernel_Raster_250.txt"))
