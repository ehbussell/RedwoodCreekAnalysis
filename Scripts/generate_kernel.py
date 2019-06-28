"""Script to generate SOD kernel from Meentemeyer."""

import os
from collections import Counter
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
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

def make_inv_cdf(params):
    """Generate inverse cdf function of kernel."""

    max_dist = optimize.root(lambda x: (meentemeyer_cdf(x, params) - 0.9999), x0=0).x[0]
    distances = np.linspace(0, max_dist, int(1000*max_dist))
    inv_cdf = interp1d(meentemeyer_cdf(distances, params), distances, fill_value="extrapolate")

    return inv_cdf


def kernel_sample(inv_cdf, num=1):
    """Generate sample(s) from Meentemeyer kernel."""

    uniform_sample = np.random.rand(int(num))

    samples = np.array(inv_cdf(uniform_sample))

    return samples


def update_counter(counter_dict, dist_samples, angle_samples, kernel_range=10, resolution=250):
    """Extract array kernel from kernel samples and update counter"""

    sample_x = (resolution/2) + (dist_samples * np.cos(angle_samples))
    sample_y = (resolution/2) + (dist_samples * np.sin(angle_samples))

    cell_x = [int(x // resolution) + kernel_range for x in sample_x]
    cell_y = [int(y // resolution) + kernel_range for y in sample_y]

    all_cells = zip(cell_x, cell_y)

    counter_dict.update(all_cells)


def generate_kernel(file_name=None, params=None):
    """Generate ASCII raster format kernel for SOD simulations."""

    if file_name is None:
        file_name = "Kernel_Raster_250.txt"

    if params is None:
        meentemeyer_params = {
            'gamma': 0.9947,
            'alpha_1': 20.57,
            'alpha_2': 9500
        }
    else:
        meentemeyer_params = {
            'gamma': params['gamma'],
            'alpha_1': params['alpha_1'],
            'alpha_2': params['alpha_2']
        }

    num_samples = int(1e9)
    samples_left = num_samples

    max_bin_size = int(1e8)
    host_raster_eroi = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "EROI_Landscape", "HostDensity.txt"
    ))
    kernel_range = max(host_raster_eroi.array.shape)
    counter_dict = Counter()

    inv_cdf = make_inv_cdf(meentemeyer_params)

    while samples_left > 0:
        nsamp = min(samples_left, max_bin_size)
        dist_samples = kernel_sample(inv_cdf, num=nsamp)
        angle_samples = np.random.rand(nsamp) * 2 * np.pi
        samples_left -= nsamp
        print("Block done")

        update_counter(counter_dict, dist_samples, angle_samples, kernel_range, 250)

    kernel_array = np.array([[counter_dict[(x, y)] / num_samples for y in range(2*kernel_range+1)]
                             for x in range(2*kernel_range+1)])
    kernel_raster = raster_tools.RasterData(
        (2*kernel_range+1, 2*kernel_range+1), array=kernel_array, cellsize=250)

    kernel_raster.to_file(os.path.join("GeneratedData", file_name))
