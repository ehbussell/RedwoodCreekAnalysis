"""Script running resolution tests for Redwood Creek analysis."""

from IPython import embed
import argparse
import json
import logging
import os
import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from RasterModel import raster_model_fitting
from RasterModel import raster_model
import raster_tools
from Scripts import generate_landscapes
from Scripts import kernels
from Scripts import fit_test
from Scripts import MainOptions
from Scripts import summarise_sims
from Scripts import optimise_control

OPTIONS = {
    'test_resolutions': [5000, 2500, 2000, 1500, 1000, 500, 250],
    'kernel_names': ["Exponential", "Cauchy"],
    'fit_method': 'SSE',

    # Kernel prior for each resolution and for each kernel
    'kernel_priors': [
        [{"beta": (0, 0.05), "scale": (0, 5)}, {"beta": (1e-6, 0.01), "scale": (1e-6, 5)}],
        [{"beta": (0, 0.05), "scale": (0, 5)}, {"beta": (1e-6, 0.01), "scale": (1e-6, 5)}],
        [{"beta": (0, 0.05), "scale": (0, 5)}, {"beta": (1e-5, 0.01), "scale": (1e-4, 5)}],
        [{"beta": (0, 0.1), "scale": (0, 5)}, {"beta": (1e-6, 0.01), "scale": (1e-3, 5)}],
        [{"beta": (0, 1), "scale": (0, 5)}, {"beta": (1e-5, 0.01), "scale": (1e-4, 5)}],
        [{"beta": (0, 2), "scale": (0, 5)}, {"beta": (1e-5, 0.1), "scale": (1e-2, 5)}],
        [{"beta": (0.05, 2), "scale": (0, 5)}, {"beta": (1e-5, 0.1), "scale": (1e-2, 5)}]
    ],

    # Kernel intialisation for each resolution and for each kernel
    'kernel_inits': [
        [{"beta": 0.001, "scale": 0.2}, {"beta": 0.0005, "scale": 0.3}],
        [{"beta": 0.005, "scale": 0.4}, {"beta": 0.005, "scale": 0.2}],
        [{"beta": 0.005, "scale": 0.5}, {"beta": 0.002, "scale": 0.2}],
        [{"beta": 0.01, "scale": 0.7}, {"beta": 0.008, "scale": 0.2}],
        [{"beta": 0.1, "scale": 1.0}, {"beta": 0.006, "scale": 0.2}],
        [{"beta": 0.5, "scale": 2.0}, {"beta": 0.03, "scale": 0.02}],
        [{"beta": 0.5, "scale": 3.5}, {"beta": 0.03, "scale": 0.02}]
    ]
}

def make_landscapes(test_resolutions, roi):
    """Generate landscapes"""

    for resolution in test_resolutions:
        landscape_name = "ROI_" + str(resolution) + "Landscape"
        if os.path.isdir(os.path.join("GeneratedData", landscape_name)):
            logging.info("%dm resoution landscape already exists", resolution)
        else:
            logging.info("Generating %dm resoution landscape", resolution)
            generate_landscapes.generate_landscape(roi, resolution, landscape_name)

def fit_landscapes(out_file, config_dict, test_resolutions, full_resolution, sim_stub,
                   append=False):
    """Fit raster model kernel parameters to simulation data."""

    if append:
        try:
            with open(out_file, "r") as fin:
                all_fit_results = json.load(fin)
        except FileNotFoundError:
            logging.warning("FitResults file not found! Starting new results.")
            all_fit_results = {}
    else:
        all_fit_results = {}

    for i, resolution in enumerate(test_resolutions):
        logging.info("Starting fit for %dm resolution.", resolution)
        landscape_name = "ROI_" + str(resolution) + "Landscape"
        raster_header = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", landscape_name, "HostNumbers.txt")).header_vals

        kernel_names = OPTIONS['kernel_names']

        sus_inf_file = os.path.join("GeneratedData", landscape_name, "RMSMask.txt")

        kernel_generators = []
        kernel_jac_generators = []
        for name in kernel_names:
            if name == "Exponential":
                kernel_generators.append(kernels.make_exponential_kernel)
                kernel_jac_generators.append(kernels.make_exponential_jac)
            elif name == "Cauchy":
                kernel_generators.append(kernels.make_cauchy_kernel)
                kernel_jac_generators.append(kernels.make_cauchy_jac)
            elif name == "ExpPower":
                kernel_generators.append(kernels.make_exp_power_kernel)
                kernel_jac_generators.append(kernels.make_exp_power_jac)
            else:
                raise ValueError("Unknown kernel name!")

        fit_method = OPTIONS['fit_method']
        if fit_method == "MLE":
            likelihood_id = config_dict['resolution_testing_options']['fit_landscapes'][
                'likelihood_id']
            if likelihood_id is None:
                save_file = os.path.join("GeneratedData", "SimulationRuns",
                                         landscape_name+"_likelihood.npz")
            else:
                save_file = os.path.join("GeneratedData", "SimulationRuns",
                                         landscape_name+"_likelihood_" + likelihood_id + ".npz")
            lik_loaded = raster_model_fitting.LikelihoodFunction.from_file(save_file)
        elif fit_method == "SSE":
            # Check if simulation summaries have been generated
            if not os.path.isfile(os.path.join(sim_stub, "summaries", landscape_name + '.h5')):
                # If not then summarise simulations at this resolution
                summarise_sims.summarise_sims(
                    target_header=raster_header, region=MainOptions.OPTIONS['roi'],
                    sim_stub=sim_stub, landscape_name=landscape_name)

        zipped_values = zip(kernel_names, kernel_generators, kernel_jac_generators)

        for j, (name, gen, jac) in enumerate(zipped_values):
            prior = OPTIONS['kernel_priors'][i][j]
            start = OPTIONS['kernel_inits'][i][j]

            primary_rate = bool("PrimaryRate" in prior)

            if fit_method == "MLE":
                opt_params, fit_output = raster_model_fitting.fit_raster_MLE(
                    data_stub=sim_stub, kernel_generator=gen, kernel_params=prior,
                    param_start=start, target_raster=raster_header, nsims=None,
                    likelihood_func=lik_loaded, kernel_jac=jac, raw_output=True,
                    primary_rate=primary_rate)
            elif fit_method == "SSE":
                model_params = {
                    'inf_rate': 1.0,
                    'control_rate': 0,
                    'coupling': np.zeros((raster_header['nrows'], raster_header['ncols'])),
                    'times': MainOptions.OPTIONS['times'],
                    'max_hosts': int(100 * np.power(resolution / full_resolution, 2)),
                    'primary_rate': 0
                }

                sim_sum_file = os.path.join(sim_stub, "summaries", landscape_name + ".h5")

                host_file = os.path.join("GeneratedData", landscape_name, "HostDensity.txt")
                s_init_file = os.path.join("GeneratedData", landscape_name,
                                           "InitialConditions_Density_S.txt")
                i_init_file = os.path.join("GeneratedData", landscape_name,
                                           "InitialConditions_Density_I.txt")

                model = raster_model.RasterModel(
                    model_params, host_density_file=host_file, initial_s_file=s_init_file,
                    initial_i_file=i_init_file)

                opt_params, fit_output = raster_model_fitting.fit_raster_SSE(
                    model=model, kernel_generator=gen, kernel_params=prior, data_path=sim_sum_file,
                    param_start=start, n_sims=None, target_header=raster_header, raw_output=True,
                    primary_rate=primary_rate, sus_file=sus_inf_file, inf_file=sus_inf_file
                )
            else:
                raise ValueError("Unrecognised fit method!")

            logging.info("Optimisation completed %s %s %s", landscape_name, name, opt_params)

            opt_params['Raw_Output'] = fit_output.__repr__()
            opt_params['Prior'] = prior
            opt_params['Initialisation'] = start

            # Save results
            if landscape_name in all_fit_results:
                all_fit_results[landscape_name][name] = opt_params
            else:
                all_fit_results[landscape_name] = {
                    name: opt_params
                }

            with open(out_file, "w") as f_out:
                json.dump(all_fit_results, f_out, indent=4)

        logging.info("Fit for %dm resolution complete.", resolution)

def test_fits(sim_stub, fit_results_file, output_id=None, control_stub=None):
    """Test metrics for quality of fit as approximate model resolution is changed."""

    if output_id is None:
        output_id = ""
    else:
        output_id = "_" + str(output_id)

    base_raster = raster_tools.RasterData.from_file(
        os.path.join("GeneratedData", "ROI_250Landscape", "HostNumbers.txt"))
    base_raster_header = base_raster.header_vals
    dimensions = (base_raster_header['nrows'], base_raster_header['ncols'])

    test_times = MainOptions.OPTIONS['times']

    # Simulation data at full resolution
    file_path = os.path.join(sim_stub, "summaries", "ROI_250Landscape.h5")
    with h5py.File(file_path, 'r') as hf:
        sim_dpcs = hf['sim_summary_I'][:]
    logging.info("Extracted simulation data at full resolution")

    with open(fit_results_file, "r") as fin:
        all_fit_results = json.load(fin)

    test_resolutions = OPTIONS['test_resolutions']

    for resolution in test_resolutions:
        # Setup fit testing structure for this landscape
        landscape_name = "ROI_" + str(resolution) + "Landscape"

        host_number_raster = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", landscape_name, "HostNumbers.txt"))

        dimensions_agg = (host_number_raster.header_vals['nrows'],
                          host_number_raster.header_vals['ncols'])
        ncells_agg = np.prod(dimensions_agg)

        # Check if simulation summaries have been generated
        if not os.path.isfile(os.path.join(sim_stub, "summaries", landscape_name + '.h5')):
            # If not then summarise simulations at this resolution
            summarise_sims.summarise_sims(
                target_header=host_number_raster.header_vals, region=MainOptions.OPTIONS['roi'],
                sim_stub=sim_stub, landscape_name=landscape_name)

        # Simulation data aggregated to ODE raster run resolution
        file_path = os.path.join(sim_stub, "summaries", landscape_name + ".h5")
        with h5py.File(file_path, 'r') as hf:
            sim_agg_dpcs = hf['sim_summary_I'][:]
        logging.info("Extracted simulation data at %dm resolution", resolution)

        # Simulation DPCs at landscape scale
        sim_land_dpcs = np.sum(sim_agg_dpcs, axis=1)

        all_sim_data = {
            "Divided": sim_dpcs,
            "Normal": sim_agg_dpcs,
            "Landscape": sim_land_dpcs
        }

        tested_fit = fit_test.TestedFit(
            landscape_name, base_raster, host_number_raster, test_times, sim_stub)

        if control_stub is not None:
            results_rogue = pd.read_csv(control_stub + "_v.csv")
            results_thin = pd.read_csv(control_stub + "_u.csv")

            # Convert 2500m non-spatial scheme to same rate at this resolution
            rogue_total = results_rogue.values[:, 1:].sum(axis=1) / ncells_agg
            thin_total = results_thin.values[:, 1:].sum(axis=1) / ncells_agg

            tested_fit.rogue_scheme = interp1d(
                results_rogue['time'], np.tile(rogue_total, (ncells_agg, 1)), kind="zero",
                fill_value="extrapolate")
            tested_fit.thin_scheme = interp1d(
                results_thin['time'], np.tile(thin_total, (ncells_agg, 1)), kind="zero",
                fill_value="extrapolate")

        kernel_names = OPTIONS['kernel_names']
        kernel_generators = []
        for name in kernel_names:
            if name == "Exponential":
                kernel_generators.append(kernels.make_exponential_kernel)
            elif name == "Cauchy":
                kernel_generators.append(kernels.make_cauchy_kernel)
            elif name == "ExpPower":
                kernel_generators.append(kernels.make_exp_power_kernel)
            else:
                raise ValueError("Unknown kernel name!")

        for kernel_name, kernel_gen in zip(kernel_names, kernel_generators):

            # Read parameter values
            opt_params = all_fit_results[landscape_name][kernel_name]
            opt_params.pop("Raw_Output", None)
            opt_params.pop("Prior", None)
            opt_params.pop("Initialisation", None)
            primary_rate = opt_params.pop('PrimaryRate', 0.0)

            kernel = kernel_gen(**opt_params)
            max_hosts = int(100 * np.power(resolution / 250, 2))

            sus_file = 'GeneratedData/' + landscape_name + '/RMSMask.txt'
            sus_raster = raster_tools.RasterData.from_file(sus_file)
            susceptibility = np.clip(sus_raster.array, 0, None).flatten()
            inf_raster = raster_tools.RasterData.from_file(sus_file)
            infectiousness = np.clip(inf_raster.array, 0, None).flatten()

            x = np.arange(ncells_agg) % dimensions_agg[1]
            y = np.array(np.arange(ncells_agg) / dimensions_agg[1], dtype=int)
            locs = np.array(list(zip(x, y)))
            dist_condensed = pdist(locs)
            distances = squareform(dist_condensed)

            coupling = kernel(distances) * infectiousness * susceptibility[:, np.newaxis]
            control_scaling = np.power(2500 / resolution, 2)

            params = {
                'inf_rate': 1.0,
                'control_rate': MainOptions.OPTIONS['control_rate'] * control_scaling,
                'coupling': coupling,
                'times': test_times,
                'max_hosts': max_hosts,
                'primary_rate': primary_rate
            }

            host_file = os.path.join("GeneratedData", landscape_name, "HostDensity.txt")
            s_init_file = os.path.join("GeneratedData", landscape_name,
                                       "InitialConditions_Density_S.txt")
            i_init_file = os.path.join("GeneratedData", landscape_name,
                                       "InitialConditions_Density_I.txt")

            approx_model = raster_model.RasterModel(
                params, host_density_file=host_file, initial_s_file=s_init_file,
                initial_i_file=i_init_file)

            no_control_tmp = approx_model.run_scheme(
                tested_fit.thin_scheme, tested_fit.rogue_scheme)

            no_control_results = np.zeros((ncells_agg, len(test_times)))
            for cell in range(ncells_agg):
                no_control_results[cell] = (no_control_tmp.results_i["Cell" + str(cell)].values)

            metric_vals, cell_data, time_data = calculate_metric(
                all_sim_data, no_control_results, host_number_raster.header_vals,
                base_raster_header)

            logging.info("Metrics for %s (no control): %s", landscape_name, metric_vals)

            tested_fit.add_kernel(
                kernel_name, kernel_gen, cell_data, metric_vals, time_data, opt_params)

        if control_stub is None:
            tested_fit.save(os.path.join("GeneratedData", "ResolutionTesting",
                                         landscape_name + "FitTest" + output_id + ".pickle"))
        else:
            tested_fit.save(os.path.join(
                "GeneratedData", "ResolutionTesting", landscape_name + "FitTest" + output_id +
                "_controlled.pickle"))

def rescale_control_rate(optimisation_stub, sim_stub, landscape_name, kernel_name):
    """Rescale control rate using optimised model"""

    file = "GeneratedData/ResolutionTesting/{}FitTest.pickle".format(landscape_name)
    with open(file, "rb") as infile:
        tested_fit = pickle.load(infile)

    optim = raster_model.RasterOptimisation(
        output_file_stub=os.path.join(optimisation_stub, 'output'),
        input_file_stub=optimisation_stub+'/')

    approx_model = tested_fit.get_model(kernel_name)

    standard_control_rate = MainOptions.OPTIONS['control_rate']

    file_path = os.path.join(sim_stub, "summaries", landscape_name + ".h5")
    with h5py.File(file_path, 'r') as hf:
        sim_dpcs = hf['sim_summary_I'][:]

    thin_scheme = interp1d(optim.results_u['time'], optim.results_u.values[:, 1:].T,
                           kind="zero", fill_value="extrapolate")

    rogue_scheme = interp1d(optim.results_v['time'], optim.results_v.values[:, 1:].T,
                            kind="zero", fill_value="extrapolate")

    times = tested_fit.test_times
    ncells = sim_dpcs.shape[1]

    def min_func(factor):
        """Function to minimise, SSE between sims and approx at aggregated scale."""

        approx_model.params['control_rate'] = factor[0] * standard_control_rate
        controlled_tmp = approx_model.run_scheme(thin_scheme=thin_scheme, rogue_scheme=rogue_scheme)

        controlled_results = np.zeros((ncells, len(times)))
        for cell in range(ncells):
            i_vals = controlled_tmp.results_i["Cell" + str(cell)].values
            controlled_results[cell, :] = i_vals

        sse = 0
        for dataset in sim_dpcs:
            sse += np.sum(np.square(controlled_results - dataset))

        # rescale for better gradient calculation
        sse = (sse - 130000000000) / 50000000000

        logging.info("For factor %s, SSE: %f", factor, sse)
        return sse

    ret = minimize(min_func, [0.7], method='Nelder-Mead')

    # Write scaling results to file
    results = {
        'control_rate_factor': ret.x[0],
        'Raw_Output': ret.__repr__()
    }
    file = os.path.join("GeneratedData", "ResolutionTesting", 'scaling_results.json')
    with open(file, "w") as outfile:
        json.dump(results, outfile, indent=4)

def calculate_metric(sim_data, model_data, agg_header, base_header):
    """Calculate RMSE metrics for at landscape, divided and normal scales."""

    cell_data = {}
    metric_vals = {}
    time_data = {}

    norm = sim_data['Normal'].shape[0] * sim_data['Normal'].shape[2]
    agg_dims = (agg_header['nrows'], agg_header['ncols'])
    div_dims = (base_header['nrows'], base_header['ncols'])

    # Aggregated scale
    errors = sim_data['Normal'] - model_data
    cell_data['Normal'] = np.sqrt(
        np.square(errors).sum(axis=0).sum(axis=1) / norm).reshape(agg_dims)
    metric_vals['Normal'] = np.sqrt(np.mean(np.square(errors)))
    time_data['Normal'] = np.sqrt(np.mean(np.square(errors), axis=(0, 1)))

    # Landscape scale
    errors = sim_data['Landscape'] - np.sum(model_data, axis=0)
    metric_vals['Landscape'] = np.sqrt(np.mean(np.square(errors)))
    time_data['Landscape'] = np.sqrt(np.mean(np.square(errors), axis=0))

    # Divided scale
    # Make mapping to aggregate full resolution simulaton cells
    agg_map = raster_tools.aggregate_cells(base_header, agg_header, generate_reverse=False,
                                           ignore_outside_target=True)
    ncells = np.power(agg_header['cellsize'] / base_header['cellsize'], 2)

    # Construct model data at divided scale
    model_div_data = np.zeros((sim_data['Divided'].shape[1], sim_data['Divided'].shape[2]))
    for div_cell in range(sim_data['Divided'].shape[1]):
        agg_cell = agg_map[div_cell]
        model_div_data[div_cell] = model_data[agg_cell] / ncells

    # Find metrics
    norm = sim_data['Divided'].shape[0] * sim_data['Divided'].shape[2]
    errors = sim_data['Divided'] - model_div_data
    cell_data['Divided'] = np.sqrt(
        np.square(errors).sum(axis=0).sum(axis=1) / norm).reshape(div_dims)
    metric_vals['Divided'] = np.sqrt(np.mean(np.square(errors)))
    time_data['Divided'] = np.sqrt(np.mean(np.square(errors), axis=(0, 1)))

    return metric_vals, cell_data, time_data

def run_no_control(sim_stub, append=False):
    """Run resolution testing (no control) - landscape generation, fitting, test fits."""

    os.makedirs("GeneratedData/ResolutionTesting", exist_ok=True)

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join("GeneratedData", "ResolutionTesting", 'res_testing.log'))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.info("Starting no_control analysis with sim_stub: %s", sim_stub)

    logging.info("Options used: %s", OPTIONS)

    make_landscapes(OPTIONS['test_resolutions'], MainOptions.OPTIONS['roi'])

    fit_landscapes(
        out_file=os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"),
        config_dict=None, test_resolutions=OPTIONS['test_resolutions'], full_resolution=250,
        sim_stub=sim_stub, append=append)

    test_fits(
        sim_stub=sim_stub,
        fit_results_file=os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"))

    logging.info("Analysis completed")

def run_with_control(sim_stub, control_stub):
    """Run resolution testing (with control) - test fits unser non-spatial strategy."""

    os.makedirs("GeneratedData/ResolutionTesting", exist_ok=True)

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join("GeneratedData", "ResolutionTesting", 'res_testing.log'))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.info("Starting with_control analysis with sim_stub: %s, and control_stub: %s",
                 sim_stub, control_stub)

    logging.info("Options used: %s", OPTIONS)

    test_fits(
        sim_stub=sim_stub,
        fit_results_file=os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"),
        control_stub=control_stub)

    logging.info("Analysis completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_stub", help="Stub to simulation output folder.")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Append to existing fit results file")

    args = parser.parse_args()

    os.makedirs("GeneratedData/ResolutionTesting", exist_ok=True)

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join("GeneratedData", "ResolutionTesting", 'res_testing.log'))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.info("Starting script with args: %r", args)

    logging.info("Options used: %s", OPTIONS)

    make_landscapes(OPTIONS['test_resolutions'], MainOptions.OPTIONS['roi'])

    fit_landscapes(
        out_file=os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"),
        config_dict=None, test_resolutions=OPTIONS['test_resolutions'], full_resolution=250,
        sim_stub=args.sim_stub, append=args.append)

    test_fits(
        sim_stub=args.sim_stub,
        fit_results_file=os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"))

    test_fits(
        sim_stub=args.sim_stub,
        fit_results_file=os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"),
        control_stub=os.path.join("GeneratedData", "Total_non_spatial", "output"))

    logging.info("Script completed")
