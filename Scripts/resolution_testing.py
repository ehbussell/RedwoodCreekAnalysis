"""Script running resolution tests for Redwood Creek analysis."""

import os
import time
import pickle
import json
import pdb
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import IndividualSimulator
from IndividualSimulator.utilities import output_data
from RasterModel import raster_model_fitting
from RasterModel import raster_model
import raster_tools
from . import generate_landscapes
from . import kernels
from . import fit_test
from . import NSWEIntervention


def run_resolution_testing(config_dict):
    """Run tests with appropriate stages."""

    print("#"*32 + "\n## Running Resolution Testing ##\n" + "#"*32)

    roi = config_dict['general_options']['roi_extent']
    eroi = config_dict['general_options']['eroi_extent']

    full_resolution = 250
    test_resolutions = config_dict['resolution_testing_options']['test_resolutions']

    run_stats = {}

    if config_dict['resolution_testing_options']['make_landscapes']['run']:
        print("\nMaking Landscapes...\n")
        time1 = time.time()
        make_landscapes(config_dict, test_resolutions, roi)
        time2 = time.time()
        run_stats["Make_Landscapes_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['make_likelihoods']['run']:
        print("\nMaking Likelihoods...\n")
        time1 = time.time()
        make_likelihoods(config_dict, test_resolutions)
        time2 = time.time()
        run_stats["Make_Likelihoods_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['fit_landscapes']['run']:
        print("\nFitting Landscapes...\n")
        time1 = time.time()
        fit_landscapes(config_dict, test_resolutions, full_resolution)
        time2 = time.time()
        run_stats["Fit_Landscapes_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['test_fits']['run']:
        print("\nTesting Fits...\n")
        time1 = time.time()
        test_fits(config_dict, test_resolutions, full_resolution)
        time2 = time.time()
        run_stats["Test_Fits_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['test_control_fits']['run']:
        print("\nTesting Fits under Control...\n")
        time1 = time.time()
        test_control_fits(config_dict, test_resolutions, full_resolution)
        time2 = time.time()
        run_stats["Test_Control_Fits_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['plot_fits']['run']:
        print("\nPlotting Fit Asessments...\n")
        time1 = time.time()
        plot_fits(config_dict, test_resolutions, full_resolution)
        time2 = time.time()
        run_stats["Plot_Fits_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['plot_control_fits']['run']:
        print("\nPlotting Control Fit Asessments...\n")
        time1 = time.time()
        plot_control_fits(config_dict, test_resolutions, full_resolution)
        time2 = time.time()
        run_stats["Plot_Control_Fits_Time"] = str(timedelta(seconds=time2-time1))

    return run_stats

def make_landscapes(config_dict, test_resolutions, roi):
    """Generate landscapes"""

    for resolution in test_resolutions:
        landscape_name = "ROI_" + str(resolution) + "Landscape"
        options = {
            'plots': config_dict['resolution_testing_options']['make_landscapes']['make_plots']
        }
        generate_landscapes.generate_landscape(roi, resolution, landscape_name, options)

def make_likelihoods(config_dict, test_resolutions):
    """Precalculate likelihood functions."""

    for resolution in test_resolutions:
        # Make likelihood function
        landscape_name = "ROI_" + str(resolution) + "Landscape"
        raster_header = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", landscape_name, "HostNumbers.txt")).header_vals

        if resolution >= 1500:
            precompute_level = "full"
        else:
            precompute_level = "partial"

        simulation_stub = os.path.join(
            "GeneratedData", "SimulationRuns",
            config_dict['resolution_testing_options']['make_likelihoods']['simulation_stub'])

        likelihood_function = raster_model_fitting.precompute_loglik(
            data_stub=simulation_stub,
            nsims=None, raster_header=raster_header, end_time=1000, ignore_outside_raster=True,
            precompute_level=precompute_level)

        save_file = os.path.join("GeneratedData", "SimulationRuns",
                                 landscape_name+"_likelihood")
        likelihood_id = config_dict[
            'resolution_testing_options']['make_likelihoods']['likelihood_id']
        if likelihood_id is not None:
            save_file += "_" + likelihood_id
        likelihood_function.save(save_file, identifier=landscape_name)

def fit_landscapes(config_dict, test_resolutions, full_resolution):
    """Fit raster model kernel parameters to simulation data."""

    if config_dict['resolution_testing_options']['fit_landscapes']['overwrite_results_file']:
        all_fit_results = {}
    else:
        filename = os.path.join(
            "GeneratedData", "RasterFits",
            config_dict['resolution_testing_options']['fit_landscapes']['results_file_name'])
        try:
            with open(filename, "r") as fin:
                all_fit_results = json.load(fin)
        except FileNotFoundError:
            all_fit_results = {}

    reuse_start = os.path.join(
        "GeneratedData", "RasterFits",
        config_dict['resolution_testing_options']['fit_landscapes']['reuse_start'])
    with open(reuse_start, "r") as fin:
        previous_results = json.load(fin)

    for resolution in test_resolutions:
        t1 = time.time()
        landscape_name = "ROI_" + str(resolution) + "Landscape"
        raster_header = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", landscape_name, "HostNumbers.txt")).header_vals

        kernel_names = config_dict['resolution_testing_options']['kernel_names']
        kernel_priors = [{} for _ in kernel_names]
        param_start = [{} for _ in kernel_names]
        if reuse_start is None:
            # Initialise from parameters given in config file
            for i, kernel in enumerate(kernel_names):
                params = config_dict['resolution_testing_options']['kernel_priors'][i].keys()
                for param in params:
                    kernel_priors[i][param] = config_dict['resolution_testing_options'][
                        'kernel_priors'][i][param]
                    param_start[i][param] = config_dict['resolution_testing_options'][
                        'kernel_init'][i][param]
                    if param == "scale":
                        kernel_priors[i][param] = kernel_priors[i][param]
                        param_start[i][param] = param_start[i][param]

        else:
            # Initialise from previous results
            for i, kernel in enumerate(kernel_names):
                kernel_priors[i] = previous_results[landscape_name][kernel]["Prior"]
                param_start[i] = previous_results[landscape_name][kernel]["Initialisation"]

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

        fit_method = config_dict['resolution_testing_options']['fit_landscapes']['fit_method']
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

        zipped_values = zip(kernel_names, kernel_generators, kernel_jac_generators,
                            kernel_priors, param_start)

        for name, gen, jac, prior, start in zipped_values:

            primary_rate = bool("PrimaryRate" in prior)

            if fit_method == "MLE":
                opt_params, fit_output = raster_model_fitting.fit_raster_MLE(
                    data_stub=os.path.join("GeneratedData", "SimulationRuns", "output"),
                    kernel_generator=gen, kernel_params=prior, param_start=start,
                    target_raster=raster_header, nsims=None, likelihood_func=lik_loaded,
                    kernel_jac=jac, raw_output=True, primary_rate=primary_rate
                )
            elif fit_method == "SSE":
                model_params = {
                    'inf_rate': 1.0,
                    'control_rate': 0,
                    'max_budget_rate': 0,
                    'coupling': None,
                    'times': np.linspace(0, 1000, 101),
                    'max_hosts': int(100 * np.power(resolution / full_resolution, 2)),
                    'primary_rate': 0
                }

                host_file = os.path.join("GeneratedData", landscape_name, "HostDensity.txt")
                s_init_file = os.path.join("GeneratedData", landscape_name,
                                           "InitialConditions_Density_S.txt")
                i_init_file = os.path.join("GeneratedData", landscape_name,
                                           "InitialConditions_Density_I.txt")

                model = raster_model.RasterModel(
                    model_params, host_density_file=host_file, initial_s_file=s_init_file,
                    initial_i_file=i_init_file)

                opt_params, fit_output = raster_model_fitting.fit_raster_SSE(
                    model=model, kernel_generator=gen, kernel_params=prior,
                    data_stub=os.path.join("GeneratedData", "SimulationRuns", "output"),
                    param_start=start, nsims=None, target_raster=raster_header, raw_output=True,
                    primary_rate=primary_rate
                )
            else:
                raise ValueError("Unrecognised fit method!")

            print("Complete. {0}, {1} kernel: {2}".format(landscape_name, name, opt_params))

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

            outfile = os.path.join(
                "GeneratedData", "RasterFits",
                config_dict['resolution_testing_options']['fit_landscapes']['results_file_name']
            )
            with open(outfile, "w") as f_out:
                json.dump(all_fit_results, f_out, indent=4)

            if config_dict['resolution_testing_options']['fit_landscapes']['make_plots']:
                plot_likelihood(landscape_name, name, gen, opt_params, lik_loaded, config_dict,
                                prior)

        t2 = time.time()
        time_taken = timedelta(seconds=t2-t1)
        print("Landscape ROI_" + str(resolution) +
              " Fitted. Time taken: {0}s".format(str(time_taken)))

def test_fits(config_dict, test_resolutions, full_resolution):
    """Assess fits of raster models to simulation data."""

    data_path = os.path.join("GeneratedData", "SimulationRuns", "output")
    base_raster = raster_tools.RasterData.from_file(
        os.path.join("GeneratedData", "ROI_250Landscape", "HostDensity.txt"))
    base_raster_header = base_raster.header_vals
    dimensions = (base_raster_header['nrows'], base_raster_header['ncols'])
    ncells = np.prod(dimensions)

    test_times = np.linspace(0, 1000, 101)

    # Simulation data at full resolution
    base_data = output_data.create_cell_data(
        data_path, target_header=base_raster_header, ignore_outside_raster=True)
    sim_dpcs = np.zeros((len(base_data), len(test_times), ncells))
    for i, dataset in enumerate(base_data):
        for cell in range(ncells):
            current_i = None
            idx = 0
            for t, _, i_state, *_ in dataset[cell]:
                while t > test_times[idx]:
                    sim_dpcs[i, idx, cell] = current_i
                    idx += 1
                    if idx > len(test_times):
                        break
                current_i = i_state
            while idx != len(test_times):
                sim_dpcs[i, idx, cell] = current_i
                idx += 1

    sim_land_dpcs = np.sum(sim_dpcs, axis=2)

    infile = os.path.join(
        "GeneratedData", "RasterFits",
        config_dict['resolution_testing_options']['test_fits']['fit_results_file_name'])
    with open(infile, "r") as fin:
        all_fit_results = json.load(fin)

    n_short_time_qa_periods = config_dict[
        'resolution_testing_options']['test_fits']['n_short_time_qa_periods']

    for resolution in test_resolutions:
        # Setup fit testing structure for this landscape
        landscape_name = "ROI_" + str(resolution) + "Landscape"

        run_raster = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", landscape_name, "HostDensity.txt"))
        run_raster_header = run_raster.header_vals
        dimensions_agg = (run_raster_header['nrows'], run_raster_header['ncols'])
        ncells_agg = np.prod(dimensions_agg)

        # Simulation data aggregated to ODE raster run resolution
        aggregated_base_data = output_data.create_cell_data(
            data_path, target_header=run_raster_header, ignore_outside_raster=True)
        sim_agg_dpcs = np.zeros((len(aggregated_base_data), len(test_times), ncells_agg))
        for i, dataset in enumerate(aggregated_base_data):
            for cell in range(ncells_agg):
                current_i = None
                idx = 0
                for t, _, i_state, *_ in dataset[cell]:
                    while t > test_times[idx]:
                        sim_agg_dpcs[i, idx, cell] = current_i
                        idx += 1
                        if idx > len(test_times):
                            break
                    current_i = i_state
                while idx != len(test_times):
                    sim_agg_dpcs[i, idx, cell] = current_i
                    idx += 1

        all_sim_data = {
            "Divided": sim_dpcs,
            "Normal": sim_agg_dpcs,
            "Landscape": sim_land_dpcs
        }

        tested_fit = fit_test.TestedFit(
            landscape_name, base_raster, run_raster, all_sim_data, test_times)

        if n_short_time_qa_periods is not None:
            tested_fit_short = fit_test.TestedFit(
                landscape_name, base_raster, run_raster, all_sim_data, test_times,
                coupled_runs=True)

        kernel_names = config_dict['resolution_testing_options']['kernel_names']
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

            print("Resolution: {0}, Kernel: {1}".format(resolution, kernel_name), opt_params)
            print("PrimaryRate: {0}".format(primary_rate))

            coupling = np.ones((ncells_agg, ncells_agg))
            kernel = kernel_gen(**opt_params)

            control_rate = 0
            max_budget_rate = 0

            for i in range(ncells_agg):
                for j in range(ncells_agg):
                    dx = abs((i % dimensions_agg[1]) - (j % dimensions_agg[1]))
                    dy = abs(int(i/dimensions_agg[1]) - int(j/dimensions_agg[1]))
                    dist = np.sqrt(dx*dx + dy*dy)
                    coupling[i, j] = kernel(dist)

            # times = np.linspace(0, 1000, 501)
            max_hosts = int(100 * np.power(resolution / full_resolution, 2))

            params = {
                'inf_rate': 1.0,
                'control_rate': control_rate,
                'max_budget_rate': max_budget_rate,
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

            # TODO change to ndarray format rather than dictionary? - easier to work with
            no_control_tmp = approx_model.run_scheme(approx_model.no_control_policy)
            no_control_results = {}
            for cell in range(ncells_agg):
                s_vals = no_control_tmp.results_s["Cell" + str(cell)].values
                i_vals = no_control_tmp.results_i["Cell" + str(cell)].values
                t_vals = no_control_tmp.results_s["time"].values
                no_control_results[cell] = np.column_stack((t_vals, s_vals, i_vals))

            no_control_metrics = calculate_metric(
                tested_fit, no_control_results, run_raster_header, base_raster_header,
                kernel_name, metric="RMSE")

            if n_short_time_qa_periods is not None:
                run_short_qa(tested_fit_short, run_raster_header, base_raster_header,
                             kernel_name, landscape_name, approx_model, n_short_time_qa_periods)

            print(no_control_metrics)

        tested_fit.save(os.path.join(
            "GeneratedData", "RasterFits", landscape_name + "FitTest_" +
            config_dict['resolution_testing_options']['test_fits']['output_id'] + ".pickle"))

        if n_short_time_qa_periods is not None:
            tested_fit_short.save(os.path.join(
                "GeneratedData", "RasterFits", landscape_name + "FitTest_" +
                config_dict['resolution_testing_options']['test_fits']['output_id'] +
                "_ShortQA.pickle"))

def test_control_fits(config_dict, test_resolutions, full_resolution):
    """Assess fits of raster models under control."""

    all_budgets = config_dict['resolution_testing_options']['test_control_fits']['budgets']
    control_rate = config_dict['resolution_testing_options']['test_control_fits'][
        'control_rate']
    test_times = np.linspace(0, 1000, 101)

    n_short_time_qa_periods = config_dict[
        'resolution_testing_options']['test_control_fits']['n_short_time_qa_periods']

    base_raster = raster_tools.RasterData.from_file(
        os.path.join("GeneratedData", "ROI_250Landscape", "HostDensity.txt"))
    base_raster_header = base_raster.header_vals
    dimensions = (base_raster_header['nrows'], base_raster_header['ncols'])
    ncells = np.prod(dimensions)

    for resolution in test_resolutions:
        # Setup fit testing structure for this landscape
        landscape_name = "ROI_" + str(resolution) + "Landscape"

        run_raster = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", landscape_name, "HostDensity.txt"))
        run_raster_header = run_raster.header_vals
        dimensions_agg = (run_raster_header['nrows'], run_raster_header['ncols'])
        ncells_agg = np.prod(dimensions_agg)

        infile = os.path.join(
            "GeneratedData", "RasterFits",
            config_dict['resolution_testing_options']['test_control_fits']['fit_results_file_name'])
        with open(infile, "r") as fin:
            all_fit_results = json.load(fin)

        kernel_names = config_dict['resolution_testing_options']['kernel_names']
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

        for budget in all_budgets:

            # Make map to low resolution cells for intervention
            sim_header = raster_tools.RasterData.from_file(
                os.path.join("GeneratedData", "EROI_Landscape", "HostNumbers.txt")).header_vals
            forward_map, reverse_map = raster_tools.aggregate_cells(
                sim_header, run_raster_header, generate_reverse=True, ignore_outside_target=True)
            # Run simulation data with intervention scheme (NSWE, at given run resolution)
            data_path = os.path.join(
                "GeneratedData", "SimulationRuns", config_dict['resolution_testing_options'][
                    'test_control_fits']['simulation_stub'] + "_controlled")
            add_options = config_dict[
                'resolution_testing_options']['test_control_fits']['simulation_add_options']
            if add_options is None:
                add_options = {}
            add_options["InterventionScripts"] = [NSWEIntervention.Intervention]
            add_options["InterventionUpdateFrequencies"] = None
            add_options["UpdateOnAllEvents"] = True
            add_options["InterventionOptions"] = [(
                budget, control_rate, forward_map, reverse_map)]
            add_options["OutputFileStub"] = data_path

            run_data = IndividualSimulator.main(
                os.path.join("InputData", "REDW_config.ini"), params_options=add_options)

            # Extract simulation data
            base_data = output_data.create_cell_data(
                data_path, target_header=base_raster_header, ignore_outside_raster=True)
            sim_dpcs = np.zeros((len(base_data), len(test_times), ncells))
            for i, dataset in enumerate(base_data):
                for cell in range(ncells):
                    current_i = None
                    idx = 0
                    for t, _, i_state, *_ in dataset[cell]:
                        while t > test_times[idx]:
                            sim_dpcs[i, idx, cell] = current_i
                            idx += 1
                            if idx > len(test_times):
                                break
                        current_i = i_state
                    while idx != len(test_times):
                        sim_dpcs[i, idx, cell] = current_i
                        idx += 1

            sim_land_dpcs = np.sum(sim_dpcs, axis=2)

            aggregated_base_data = output_data.create_cell_data(
                data_path, target_header=run_raster_header, ignore_outside_raster=True)
            sim_agg_dpcs = np.zeros((len(aggregated_base_data), len(test_times), ncells_agg))
            for i, dataset in enumerate(aggregated_base_data):
                for cell in range(ncells_agg):
                    current_i = None
                    idx = 0
                    for t, _, i_state, *_ in dataset[cell]:
                        while t > test_times[idx]:
                            sim_agg_dpcs[i, idx, cell] = current_i
                            idx += 1
                            if idx > len(test_times):
                                break
                        current_i = i_state
                    while idx != len(test_times):
                        sim_agg_dpcs[i, idx, cell] = current_i
                        idx += 1

            all_sim_data = {
                "Divided": sim_dpcs,
                "Normal": sim_agg_dpcs,
                "Landscape": sim_land_dpcs
            }

            tested_fit = fit_test.TestedFit(
                landscape_name, base_raster, run_raster, all_sim_data, test_times)

            if n_short_time_qa_periods is not None:
                tested_fit_short = fit_test.TestedFit(
                    landscape_name, base_raster, run_raster, all_sim_data, test_times,
                    coupled_runs=True)

            for kernel_name, kernel_gen in zip(kernel_names, kernel_generators):

                # Read parameter values
                opt_params = all_fit_results[landscape_name][kernel_name]
                opt_params.pop("Raw_Output", None)
                opt_params.pop("Prior", None)
                opt_params.pop("Initialisation", None)
                primary_rate = opt_params.pop('PrimaryRate', 0.0)

                print("Resolution: {0}, Kernel: {1}".format(resolution, kernel_name), opt_params)
                print("PrimaryRate: {0}".format(primary_rate))

                coupling = np.ones((ncells_agg, ncells_agg))
                kernel = kernel_gen(**opt_params)

                for i in range(ncells_agg):
                    for j in range(ncells_agg):
                        dx = abs((i % dimensions_agg[1]) - (j % dimensions_agg[1]))
                        dy = abs(int(i/dimensions_agg[1]) - int(j/dimensions_agg[1]))
                        dist = np.sqrt(dx*dx + dy*dy)
                        coupling[i, j] = kernel(dist)

                # times = np.linspace(0, 1000, 501)
                max_hosts = int(100 * np.power(resolution / full_resolution, 2))

                params = {
                    'inf_rate': 1.0,
                    'control_rate': control_rate,
                    'max_budget_rate': budget,
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

                # Calculate quality metrics with control
                # Run policy on raster model (with control, prioritised by N->S, E->W)
                control_scheme = NSWEIntervention.RasterControlScheme(ncells_agg, budget)
                control_tmp = approx_model.run_scheme(control_scheme.control_scheme)
                control_results = {}
                for cell in range(ncells_agg):
                    s_vals = control_tmp.results_s["Cell" + str(cell)].values
                    i_vals = control_tmp.results_i["Cell" + str(cell)].values
                    t_vals = control_tmp.results_s["time"].values
                    control_results[cell] = np.column_stack((t_vals, s_vals, i_vals))
                # Calculate quality metrics
                control_metrics_tmp = calculate_metric(
                    tested_fit, control_results, run_raster_header, base_raster_header,
                    kernel_name, metric="RMSE")

                if n_short_time_qa_periods is not None:
                    run_short_qa(tested_fit_short, run_raster_header, base_raster_header,
                                 kernel_name, landscape_name, approx_model, n_short_time_qa_periods,
                                 control_scheme=control_scheme.control_scheme)

            tested_fit.save(os.path.join(
                "GeneratedData", "RasterFits", landscape_name + "ControlFitTest_Budget" +
                str(budget) + "_" +
                config_dict['resolution_testing_options']['test_control_fits']['output_id'] +
                ".pickle"))

            if n_short_time_qa_periods is not None:
                tested_fit_short.save(os.path.join(
                    "GeneratedData", "RasterFits", landscape_name + "ControlFitTest_Budget" +
                    str(budget) + "_" +
                    config_dict['resolution_testing_options']['test_control_fits']['output_id'] +
                    "_ShortQA.pickle"))


def plot_fits(config_dict, test_resolutions, full_resolution):
    """Plot fit assessments to test resolutions."""

    time_qa_plots = config_dict['resolution_testing_options']['plot_fits']['time_qa_plots']
    include_short_qa = config_dict[
        'resolution_testing_options']['plot_fits']['include_short_time_qa']

    save_path = os.path.join(
        "Figures", "ResolutionTesting_" +
        config_dict['resolution_testing_options']['plot_fits']['output_name'], "NoControl")

    os.makedirs(save_path, exist_ok=True)

    kernel_names = config_dict['resolution_testing_options']['kernel_names']

    # Combine data
    metric_scales = ["Divided", "Normal", "Landscape"]
    all_data = {kernel: {metric: [] for metric in metric_scales} for kernel in kernel_names}
    short_data = {kernel: {metric: [] for metric in metric_scales} for kernel in kernel_names}
    for resolution in test_resolutions:
        nhosts = int(100 * np.power(resolution / full_resolution, 2))
        landscape_name = "ROI_" + str(resolution) + "Landscape"
        file_name = os.path.join(
            "GeneratedData", "RasterFits", landscape_name + "FitTest_" +
            config_dict['resolution_testing_options']['plot_fits']['test_output_id'] +
            ".pickle")
        with open(file_name, "rb") as infile:
            tested_fit = pickle.load(infile)
        nhosts_land = (100 * tested_fit.base_raster.header_vals['nrows'] *
                       tested_fit.base_raster.header_vals['ncols'])

        if include_short_qa:
            file_name = os.path.join(
                "GeneratedData", "RasterFits", landscape_name + "FitTest_" +
                config_dict['resolution_testing_options']['plot_fits']['test_output_id'] +
                "_ShortQA.pickle")
            with open(file_name, "rb") as infile:
                tested_fit_short = pickle.load(infile)

        for kernel in kernel_names:
            all_data[kernel]["Divided"].append(
                tested_fit.metric_data[kernel]["Divided"] / 100)
            all_data[kernel]["Normal"].append(
                tested_fit.metric_data[kernel]["Normal"] / nhosts)
            all_data[kernel]["Landscape"].append(
                tested_fit.metric_data[kernel]["Landscape"] / nhosts_land)

            if include_short_qa:
                short_data[kernel]["Divided"].append(
                    tested_fit_short.metric_data[kernel]["Divided"] / 100)
                short_data[kernel]["Normal"].append(
                    tested_fit_short.metric_data[kernel]["Normal"] / nhosts)
                short_data[kernel]["Landscape"].append(
                    tested_fit_short.metric_data[kernel]["Landscape"] / nhosts_land)

        if time_qa_plots:
            test_times = tested_fit.test_times
            time_data = tested_fit.time_data
            if include_short_qa:
                test_times_short = tested_fit_short.test_times
                time_data_short = tested_fit_short.time_data
            for metric in metric_scales:
                if metric == "Divided":
                    host_factor = 100
                elif metric == "Normal":
                    host_factor = nhosts
                elif metric == "Landscape":
                    host_factor = nhosts_land
                else:
                    raise ValueError("Unknown metric scale!")

                fig = plt.figure()
                ax = fig.add_subplot(111)
                for i, kernel in enumerate(kernel_names):
                    ax.plot(test_times, time_data[kernel][metric] / host_factor, '-',
                            color="C{}".format(i), alpha=0.6, label=kernel)
                    if include_short_qa:
                        ax.plot(test_times_short, time_data_short[kernel][metric] / host_factor,
                                '--', color="C{}".format(i), alpha=0.6)
                ax.legend()
                ax.set_title(metric + " Metric")
                ax.set_xlabel("Time")
                ax.set_ylabel("RMSE as Proportion of Cell")
                fig.savefig(os.path.join(
                    save_path, "ResolutionTesting_TimeQA_" + metric + "_" +
                    config_dict['resolution_testing_options']['plot_fits']['output_name'] +
                    "_" + str(resolution) + ".png"))
                plt.close(fig)

    # Make plots for each kernel
    for kernel in kernel_names:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, metric in enumerate(metric_scales):
            ax.plot(test_resolutions, all_data[kernel][metric], 'o-', alpha=0.6, label=metric,
                    color="C{}".format(i))
            if include_short_qa:
                ax.plot(test_resolutions, short_data[kernel][metric], 'o--', alpha=0.6,
                        color="C{}".format(i))
        ax.legend()
        ax.set_title(kernel + " Kernel")
        ax.set_xlabel("Resolution / m")
        ax.set_ylabel("RMSE as Proportion of Cell")
        fig.savefig(os.path.join(
            save_path, "ResolutionTesting_" + kernel + "_" +
            config_dict['resolution_testing_options']['plot_fits']['output_name'] + ".png"))
        plt.close(fig)

    # Make plots for each metric
    for metric in metric_scales:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, kernel in enumerate(kernel_names):
            ax.plot(test_resolutions, all_data[kernel][metric], 'o-', alpha=0.6, label=kernel,
                    color="C{}".format(i))
            if include_short_qa:
                ax.plot(test_resolutions, short_data[kernel][metric], 'o--', alpha=0.6,
                        color="C{}".format(i))
        ax.legend()
        ax.set_title(metric + " Metric")
        ax.set_xlabel("Resolution / m")
        ax.set_ylabel("RMSE as Proportion of Cell")
        fig.savefig(os.path.join(
            save_path, "ResolutionTesting_" + metric + "_" +
            config_dict['resolution_testing_options']['plot_fits']['output_name'] + ".png"))
        plt.close(fig)

def plot_control_fits(config_dict, test_resolutions, full_resolution):
    """Plot control fit assessments to test resolutions over control budget range."""

    all_budgets = config_dict['resolution_testing_options']['plot_control_fits']['budgets']

    time_qa_plots = config_dict['resolution_testing_options']['plot_control_fits']['time_qa_plots']
    include_short_qa = config_dict[
        'resolution_testing_options']['plot_control_fits']['include_short_time_qa']

    save_path = os.path.join(
        "Figures", "ResolutionTesting_" +
        config_dict['resolution_testing_options']['plot_control_fits']['output_name'],
        "WithControl")

    os.makedirs(save_path, exist_ok=True)

    kernel_names = config_dict['resolution_testing_options']['kernel_names']

    # Combine data
    metric_scales = ["Divided", "Normal", "Landscape"]
    all_data = {kernel: {metric: {resolution: [] for resolution in test_resolutions}
                         for metric in metric_scales} for kernel in kernel_names}
    short_data = {kernel: {metric: {resolution: [] for resolution in test_resolutions}
                           for metric in metric_scales} for kernel in kernel_names}
    for resolution in test_resolutions:
        nhosts = int(100 * np.power(resolution / full_resolution, 2))
        landscape_name = "ROI_" + str(resolution) + "Landscape"

        file_name = os.path.join(
            "GeneratedData", "RasterFits", landscape_name + "FitTest_" +
            config_dict['resolution_testing_options']['plot_control_fits']['test_output_id'] +
            ".pickle")
        with open(file_name, "rb") as infile:
            tested_fit = pickle.load(infile)
            nhosts_land = (100 * tested_fit.base_raster.header_vals['nrows'] *
                           tested_fit.base_raster.header_vals['ncols'])

        if include_short_qa:
            file_name = os.path.join(
                "GeneratedData", "RasterFits", landscape_name + "FitTest_" +
                config_dict['resolution_testing_options']['plot_control_fits']['test_output_id'] +
                "_ShortQA.pickle")
            with open(file_name, "rb") as infile:
                tested_fit_short = pickle.load(infile)

        if time_qa_plots:
            all_time_data = {kernel: {metric: [] for metric in metric_scales}
                             for kernel in kernel_names}
            all_test_times = {kernel: {metric: [] for metric in metric_scales}
                              for kernel in kernel_names}
            if include_short_qa:
                all_time_data_short = {kernel: {metric: [] for metric in metric_scales}
                                       for kernel in kernel_names}
                all_test_times_short = {kernel: {metric: [] for metric in metric_scales}
                                        for kernel in kernel_names}

        for kernel in kernel_names:
            all_data[kernel]["Divided"][resolution].append(
                tested_fit.metric_data[kernel]["Divided"] / 100)
            all_data[kernel]["Normal"][resolution].append(
                tested_fit.metric_data[kernel]["Normal"] / nhosts)
            all_data[kernel]["Landscape"][resolution].append(
                tested_fit.metric_data[kernel]["Landscape"] / nhosts_land)

            if include_short_qa:
                short_data[kernel]["Divided"][resolution].append(
                    tested_fit_short.metric_data[kernel]["Divided"] / 100)
                short_data[kernel]["Normal"][resolution].append(
                    tested_fit_short.metric_data[kernel]["Normal"] / nhosts)
                short_data[kernel]["Landscape"][resolution].append(
                    tested_fit_short.metric_data[kernel]["Landscape"] / nhosts_land)

            if time_qa_plots:
                test_times = tested_fit.test_times
                time_data = tested_fit.time_data
                if include_short_qa:
                    test_times_short = tested_fit_short.test_times
                    time_data_short = tested_fit_short.time_data
                for metric in metric_scales:
                    if metric == "Divided":
                        host_factor = 100
                    elif metric == "Normal":
                        host_factor = nhosts
                    elif metric == "Landscape":
                        host_factor = nhosts_land
                    else:
                        raise ValueError("Unknown metric scale!")

                    all_test_times[kernel][metric].append(test_times)
                    all_time_data[kernel][metric].append(
                        time_data[kernel][metric] / host_factor)
                    if include_short_qa:
                        all_test_times_short[kernel][metric].append(test_times_short)
                        all_time_data_short[kernel][metric].append(
                            time_data_short[kernel][metric] / host_factor)

        for budget in all_budgets:
            file_name = os.path.join(
                "GeneratedData", "RasterFits", landscape_name + "ControlFitTest_Budget" +
                str(budget) + "_" +
                config_dict['resolution_testing_options']['plot_control_fits']['test_output_id'] +
                ".pickle")
            with open(file_name, "rb") as infile:
                tested_fit = pickle.load(infile)
            nhosts_land = (100 * tested_fit.base_raster.header_vals['nrows'] *
                           tested_fit.base_raster.header_vals['ncols'])

            if include_short_qa:
                file_name = os.path.join(
                    "GeneratedData", "RasterFits", landscape_name + "ControlFitTest_Budget" +
                    str(budget) + "_" +
                    config_dict['resolution_testing_options']['plot_control_fits']['test_output_id']
                    + "_ShortQA.pickle")
                with open(file_name, "rb") as infile:
                    tested_fit_short = pickle.load(infile)

            for kernel in kernel_names:
                all_data[kernel]["Divided"][resolution].append(
                    tested_fit.metric_data[kernel]["Divided"] / 100)
                all_data[kernel]["Normal"][resolution].append(
                    tested_fit.metric_data[kernel]["Normal"] / nhosts)
                all_data[kernel]["Landscape"][resolution].append(
                    tested_fit.metric_data[kernel]["Landscape"] / nhosts_land)

                if include_short_qa:
                    short_data[kernel]["Divided"][resolution].append(
                        tested_fit_short.metric_data[kernel]["Divided"] / 100)
                    short_data[kernel]["Normal"][resolution].append(
                        tested_fit_short.metric_data[kernel]["Normal"] / nhosts)
                    short_data[kernel]["Landscape"][resolution].append(
                        tested_fit_short.metric_data[kernel]["Landscape"] / nhosts_land)

                if time_qa_plots:
                    test_times = tested_fit.test_times
                    time_data = tested_fit.time_data
                    if include_short_qa:
                        test_times_short = tested_fit_short.test_times
                        time_data_short = tested_fit_short.time_data
                    for metric in metric_scales:
                        if metric == "Divided":
                            host_factor = 100
                        elif metric == "Normal":
                            host_factor = nhosts
                        elif metric == "Landscape":
                            host_factor = nhosts_land
                        else:
                            raise ValueError("Unknown metric scale!")

                        all_test_times[kernel][metric].append(test_times)
                        all_time_data[kernel][metric].append(
                            time_data[kernel][metric] / host_factor)
                        if include_short_qa:
                            all_test_times_short[kernel][metric].append(test_times_short)
                            all_time_data_short[kernel][metric].append(
                                time_data_short[kernel][metric] / host_factor)

        if time_qa_plots:
            jet = plt.get_cmap('viridis')
            cNorm = mpl.colors.Normalize(vmin=0, vmax=max(all_budgets))
            scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=jet)
            for metric in metric_scales:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                kernel = "Exponential"
                ax.plot(all_test_times[kernel][metric][0], all_time_data[kernel][metric][0], '-',
                        color=scalarMap.to_rgba(0), alpha=0.6, label="Budget={}".format(0))
                if include_short_qa:
                    ax.plot(all_test_times_short[kernel][metric][0],
                            all_time_data_short[kernel][metric][0], '--',
                            color=scalarMap.to_rgba(0), alpha=0.6)
                for i, budget in enumerate(all_budgets):
                    ax.plot(all_test_times[kernel][metric][i+1], all_time_data[kernel][metric][i+1],
                            '-', color=scalarMap.to_rgba(all_budgets[i]), alpha=0.6,
                            label="Budget={}".format(budget))
                    if include_short_qa:
                        ax.plot(all_test_times_short[kernel][metric][i+1],
                                all_time_data_short[kernel][metric][i+1], '--',
                                color=scalarMap.to_rgba(all_budgets[i]), alpha=0.6)
                ax.legend()
                ax.set_title(metric + " Metric")
                ax.set_xlabel("Time")
                ax.set_ylabel("RMSE as Proportion of Cell")
                fig.tight_layout()
                fig.savefig(os.path.join(
                    save_path, "ResolutionTesting_TimeQA_" + metric + "_" +
                    config_dict['resolution_testing_options']['plot_control_fits']['output_name'] +
                    "_" + str(resolution) + ".png"))
                plt.close(fig)

    all_budgets.insert(0, 0)

    # Make plots for each metric
    for resolution in test_resolutions:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        kernel = "Exponential"
        for i, metric in enumerate(metric_scales):
            ax.plot(all_budgets, all_data[kernel][metric][resolution], 'o-', alpha=0.6,
                    label=metric, color="C{}".format(i))
            if include_short_qa:
                ax.plot(all_budgets, short_data[kernel][metric][resolution], 'o--', alpha=0.6,
                        color="C{}".format(i))
        ax.legend()
        ax.set_title("Resolution: " + str(resolution))
        ax.set_xlabel("Budget")
        ax.set_ylabel("RMSE as Proportion of Cell")
        fig.tight_layout()
        fig.savefig(os.path.join(
            save_path, "ResolutionTesting_" + str(resolution) + "_" +
            config_dict['resolution_testing_options']['plot_control_fits']['output_name'] + ".png"))
        plt.close(fig)


def plot_likelihood(landscape_name, name, gen, opt_params, lik_loaded, config_dict, prior):
    """Generate plots of likelihood surfaces."""

    recalc_plot_vals = config_dict[
        'resolution_testing_options']['fit_landscapes']['recalculate_plot_values']

    all_params = {}
    param_names = []
    for param, value in opt_params.items():
        if param in ["Raw_Output", "Prior", "Initialisation"]:
            continue
        elif param == "beta":
            all_params[param] = np.logspace(-8, np.log10(prior[param][1]), 100)
        else:
            all_params[param] = np.linspace(*prior[param], 100)
        param_names.append(param)
    param_names.sort()

    if len(param_names) == 2:
        if recalc_plot_vals:

            all_meshed = np.meshgrid(*[all_params[param] for param in param_names])

            loglik = np.zeros_like(all_meshed[0])
            for i in range(all_meshed[0].shape[0]):
                for j in range(all_meshed[0].shape[1]):
                    param_values = {
                        param: all_meshed[k][i, j] for k, param in enumerate(param_names)}
                    loglik[i, j] = lik_loaded.eval_loglik(gen(**param_values))[0]

            filename = os.path.join("GeneratedData", "RasterFits",
                                    landscape_name + name + "LikMapValues")
            np.savez(filename, loglik=loglik,
                     **{param: all_meshed[k] for k, param in enumerate(param_names)})

        else:
            filename = os.path.join("GeneratedData", "RasterFits",
                                    landscape_name + name + "LikMapValues.npz")
            value_dict = np.load(filename)
            loglik = value_dict['loglik']
            all_meshed = [value_dict[param] for param in param_names]

        cmap, cnorm = _map_loglik(loglik, config_dict)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(*all_meshed, loglik, cmap=cmap, norm=cnorm)
        ax.plot(*[opt_params[param] for param in param_names], "x")
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        if param_names[0] == "beta":
            ax.set_xscale("log")
        elif param_names[1] == "beta":
            ax.set_yscale("log")
        fig.colorbar(mesh, ax=ax)
        fig.savefig(os.path.join("Figures", landscape_name + name + "LikMap.png"))

    if len(param_names) == 3:
        for fixed_param_name in param_names:
            unfixed_param_names = [x for x in param_names if x != fixed_param_name]

            if recalc_plot_vals:
                all_meshed = np.meshgrid(*[all_params[param] for param in unfixed_param_names])

                loglik = np.zeros_like(all_meshed[0])
                for i in range(all_meshed[0].shape[0]):
                    for j in range(all_meshed[0].shape[1]):
                        param_values = {param: all_meshed[k][i, j]
                                        for k, param in enumerate(unfixed_param_names)}
                        param_values[fixed_param_name] = opt_params[fixed_param_name]
                        loglik[i, j] = lik_loaded.eval_loglik(gen(**param_values))[0]

                filename = os.path.join("GeneratedData", "RasterFits", landscape_name + name +
                                        "LikMap_" + fixed_param_name + "Values")
                np.savez(filename, loglik=loglik,
                         **{param: all_meshed[k] for k, param in enumerate(unfixed_param_names)})

            else:
                filename = os.path.join("GeneratedData", "RasterFits", landscape_name + name +
                                        "LikMap_" + fixed_param_name + "Values.npz")
                value_dict = np.load(filename)
                loglik = value_dict['loglik']
                all_meshed = [value_dict[param] for param in unfixed_param_names]

            cmap, cnorm = _map_loglik(loglik, config_dict)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            mesh = ax.pcolormesh(*all_meshed, loglik, cmap=cmap, norm=cnorm)
            ax.plot(*[opt_params[param] for param in unfixed_param_names], "x")
            ax.set_xlabel(unfixed_param_names[0])
            ax.set_ylabel(unfixed_param_names[1])
            if unfixed_param_names[0] == "beta":
                ax.set_xscale("log")
            elif unfixed_param_names[1] == "beta":
                ax.set_yscale("log")
            fig.colorbar(mesh, ax=ax)
            fig.savefig(os.path.join(
                "Figures", landscape_name + name + "LikMap_" + fixed_param_name + ".png"))

def _map_loglik(loglik, config_dict):

    loglik = np.ma.masked_invalid(loglik)

    vmin = np.min(loglik)
    vmax = np.max(loglik)
    print(vmin, vmax)
    linthresh = config_dict['resolution_testing_options']['fit_landscapes']['plot_linthresh']
    linscale = ((np.log10(np.abs(vmax)) - np.log10(linthresh)) +
                (np.log10(np.abs(vmin)) - np.log10(linthresh))) / 18

    norm = mpl.colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
    thresh_neg = norm(-linthresh)
    thresh_pos = norm(linthresh)

    colors1 = plt.cm.get_cmap("RdBu_r")(np.linspace(0, 0.5, 128))
    colors1 = list(zip(np.linspace(0, thresh_neg, 128), colors1))
    colors2 = plt.cm.get_cmap("RdBu_r")(np.linspace(0.5, 1, 128))
    colors2 = list(zip(np.linspace(thresh_pos, 1.0, 128), colors2))
    colors1 += [(thresh_neg, plt.cm.get_cmap("RdBu_r")(0.5)),
                (thresh_pos, plt.cm.get_cmap("RdBu_r")(0.5))]
    colors1 += colors2
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', colors1)

    return cmap, norm

def calculate_metric(tested_fit, run_data, run_raster_header, base_raster_header, kernel_name,
                     metric="RMSE", couple_runs=False):
    """Calculate fit quality metric for run_data, using base_data as simulation comparison."""

    # TODO change format to use ndarray instead.

    if not couple_runs:
        run_data = [run_data] * tested_fit.sim_data["Normal"].shape[0]
    run_data_div = []
    run_data_total = []

    for coupled_run in run_data:

        run_data_div.append({})

        run_data_total.append(np.zeros_like(coupled_run[0]))
        run_data_total[-1][:, 0] = coupled_run[0][:, 0]

        forward_map, back_map = raster_tools.aggregate_cells(
            base_raster_header, run_raster_header, generate_reverse=True,
            ignore_outside_target=True)
        n_cells = np.power(run_raster_header['cellsize']/base_raster_header['cellsize'], 2)

        for agg_cell, run in coupled_run.items():
            cells = back_map[agg_cell]
            for cell in cells:
                run_data_div[-1][cell] = np.column_stack((
                    run[:, 0],
                    np.divide(run[:, 1], n_cells),
                    np.divide(run[:, 2], n_cells)))

                run_data_total[-1][:, 1] += np.divide(run[:, 1], n_cells)
                run_data_total[-1][:, 2] += np.divide(run[:, 2], n_cells)

        if not couple_runs:
            run_data_div = run_data_div * tested_fit.sim_data["Normal"].shape[0]
            run_data_total = run_data_total * tested_fit.sim_data["Normal"].shape[0]
            break

    all_run_data = {
        "Divided": run_data_div,
        "Normal": run_data,
        "Landscape": run_data_total
    }

    if metric == "RMSE":
        metric_vals, cell_data, time_data = calculate_rmse(
            tested_fit, all_run_data, run_raster_header, time_data=True, verbose=False)
    else:
        raise ValueError("Metric not recognised!")

    if not couple_runs:
        all_run_data = {
            "Divided": run_data_div[0],
            "Normal": run_data[0],
            "Landscape": run_data_total[0]
        }

    tested_fit.add_kernel(kernel_name, all_run_data, cell_data, metric_vals, time_data)

    return metric_vals

def calculate_rmse(tested_fit, all_run_data, run_raster_header, time_data=False, verbose=True):
    """Calculate RMSE metrics at divided, normal and whole landscape scales."""

    test_times = tested_fit.test_times

    nRuns = tested_fit.sim_data["Normal"].shape[0]

    run_data_div = all_run_data['Divided']
    run_data = all_run_data['Normal']
    run_data_landscape = all_run_data['Landscape']

    nrows = run_raster_header['nrows']
    ncols = run_raster_header['ncols']
    nCells = nrows * ncols

    # Find correct time points in run_data
    all_times = run_data[0][0][:, 0]
    time_idxs = np.nonzero([x in test_times for x in all_times])[0]
    # TODO raise error if a test time is not included

    sse = {i: 0 for i in range(nCells)}

    if time_data:
        time_sses = np.zeros(len(time_idxs))
        time_sses_div = np.zeros(len(time_idxs))
        time_sses_land = np.zeros(len(time_idxs))

    # Aggregated rmse
    for (i, dataset), coupled_run in zip(enumerate(tested_fit.sim_data["Normal"]), run_data):
        for cell in range(nCells):
            # Find sum square error in each cell
            errors = [dataset[j, cell] - coupled_run[cell][idx, 2]
                      for j, idx in enumerate(time_idxs)]
            if time_data:
                time_sses += np.square(errors)
            sse_tmp = np.sum(np.square(errors))
            sse[cell] += sse_tmp

        if verbose:
            print("Run {0} complete.".format(i))

    cell_rmses = [np.sqrt(sse[i] / (len(test_times)*nRuns)) for i in range(nCells)]
    cell_rmses = np.reshape(cell_rmses, (nrows, ncols))

    agg_rmse = np.sqrt(np.sum([sse[i] for i in range(nCells)]) / (len(time_idxs)*nCells*nRuns))

    if time_data:
        time_sses = np.sqrt(time_sses / (nRuns*nCells))

    # Divided rmse
    nCells_div = tested_fit.sim_data["Divided"][0].shape[1]
    nrows_div = tested_fit.base_raster.header_vals['nrows']
    ncols_div = tested_fit.base_raster.header_vals['ncols']

    sse = {i: 0 for i in range(nCells_div)}

    div_rmse = None
    cell_rmses_div = None
    for (i, dataset), coupled_run in zip(enumerate(tested_fit.sim_data["Divided"]), run_data_div):
        for cell in range(nCells_div):
            # Find sum square error in each cell
            errors = [dataset[j, cell] - coupled_run[cell][idx, 2]
                      for j, idx in enumerate(time_idxs)]
            if time_data:
                time_sses_div += np.square(errors)
            sse_tmp = np.sum(np.square(errors))
            sse[cell] += sse_tmp

        if verbose:
            print("Run {0} complete.".format(i))

    cell_rmses_div = [np.sqrt(sse[i] / (len(test_times)*nRuns)) for i in range(nCells_div)]
    cell_rmses_div = np.reshape(cell_rmses_div, (nrows_div, ncols_div))

    div_rmse = np.sqrt(np.sum([sse[i] for i in range(nCells_div)]) / (
        len(time_idxs)*nCells_div*nRuns))

    if time_data:
        time_sses_div = np.sqrt(time_sses_div / (nRuns*nCells_div))

    # Landscape rmse
    errors = [[tested_fit.sim_data["Landscape"][i, j] - run_data_landscape[i][idx, 2]
               for j, idx in enumerate(time_idxs)] for i in range(nRuns)]
    if time_data:
        time_sses_land = np.sqrt(np.sum(np.square(errors), axis=0) / nRuns)
    total_rmse = np.sqrt(np.sum(np.square(errors)) / (len(time_idxs)*nRuns))

    metric_data = {
        "Divided": div_rmse,
        "Normal": agg_rmse,
        "Landscape": total_rmse
    }

    cell_data = {
        "Divided": cell_rmses_div,
        "Normal": cell_rmses
    }

    if time_data:
        time_data = {
            "Divided": time_sses_div,
            "Normal": time_sses,
            "Landscape": time_sses_land
        }

        return metric_data, cell_data, time_data

    return metric_data, cell_data

def run_short_qa(tested_fit, run_raster_header, base_raster_header, kernel_name, landscape_name,
                 approx_model, n_periods, control_scheme=None):
    """Calculate metrics for short time period quality analysis"""

    if control_scheme is None:
        control_scheme = approx_model.no_control_policy

    all_test_times = np.array_split(tested_fit.test_times, n_periods)
    start_idxs = [x[0] for x in np.array_split(range(len(tested_fit.test_times)), n_periods)]

    nsims = len(tested_fit.sim_data["Normal"])

    host_density_file = os.path.join("GeneratedData", landscape_name, "HostDensity.txt")
    initial_s_file = os.path.join("GeneratedData", landscape_name,
                                  "InitialConditions_Density_ShortTime_S.txt")
    initial_i_file = os.path.join("GeneratedData", landscape_name,
                                  "InitialConditions_Density_ShortTime_I.txt")

    host_raster = raster_tools.RasterData.from_file(host_density_file)

    raster_size = (run_raster_header['nrows'], run_raster_header['ncols'])
    raster_llcorner = (run_raster_header['xllcorner'], run_raster_header['yllcorner'])
    cellsize = run_raster_header['cellsize']
    initial_s_raster = raster_tools.RasterData(raster_size, raster_llcorner, cellsize)
    initial_i_raster = raster_tools.RasterData(raster_size, raster_llcorner, cellsize)

    all_run_data = [{cell: np.empty((0, 3)) for cell in range(np.prod(raster_size))}
                    for _ in range(nsims)]

    for start_idx, test_times in zip(start_idxs, all_test_times):
        approx_model.params['times'] = test_times
        for sim_num, sim_data in enumerate(tested_fit.sim_data["Normal"]):

            for i in range(raster_size[0]):
                for j in range(raster_size[1]):
                    cell_id = j + raster_size[1]*i
                    cell_num_i = sim_data[start_idx, cell_id] / (
                        host_raster.array[i, j]*approx_model.params['max_hosts'])
                    initial_i_raster.array[i, j] = cell_num_i
                    initial_s_raster.array[i, j] = 1-initial_i_raster.array[i, j]

            initial_s_raster.to_file(initial_s_file)
            initial_i_raster.to_file(initial_i_file)
            approx_model.set_init_state(host_density_file, initial_s_file, initial_i_file)

            no_control_tmp = approx_model.run_scheme(control_scheme)
            for cell in range(np.prod(raster_size)):
                s_vals = no_control_tmp.results_s["Cell" + str(cell)].values
                i_vals = no_control_tmp.results_i["Cell" + str(cell)].values
                t_vals = no_control_tmp.results_s["time"].values
                all_run_data[sim_num][cell] = np.vstack((
                    all_run_data[sim_num][cell],
                    np.column_stack((t_vals, s_vals, i_vals))))

    os.remove(initial_s_file)
    os.remove(initial_i_file)

    return calculate_metric(tested_fit, all_run_data, run_raster_header, base_raster_header,
                            kernel_name, couple_runs=True)
