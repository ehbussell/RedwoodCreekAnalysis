"""Script running resolution tests for Redwood Creek analysis."""

import os
import time
import pickle
import json
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import timedelta

from IndividualSimulator.utilities import output_data
from RasterModel import raster_model_fitting
from RasterModel import raster_model
import raster_tools
from . import generate_landscapes
from . import kernels
from . import fit_test


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
        for resolution in test_resolutions:
            # Generate landscape
            landscape_name = "ROI_" + str(resolution) + "Landscape"
            options = {
                'plots': config_dict['resolution_testing_options']['make_landscapes']['make_plots']
            }
            generate_landscapes.generate_landscape(roi, resolution, landscape_name, options)
        
        time2 = time.time()
        run_stats["Make_Landscapes_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['make_likelihoods']['run']:
        print("\nMaking Likelihoods...\n")
        time1 = time.time()
        for resolution in test_resolutions:
            # Make likelihood function
            landscape_name = "ROI_" + str(resolution) + "Landscape"
            raster_header = raster_tools.RasterData.from_file(
                os.path.join("GeneratedData", landscape_name, "HostNumbers.txt")).header_vals

            if resolution >= 1500:
                precompute_level = "full"
            else:
                precompute_level = "partial"

            likelihood_function = raster_model_fitting.precompute_loglik(
                data_stub=os.path.join("GeneratedData", "SimulationRuns", "output"),
                nsims=None, raster_header=raster_header, end_time=1000, ignore_outside_raster=True,
                precompute_level=precompute_level)

            save_file = os.path.join("GeneratedData", "SimulationRuns",
                                     landscape_name+"_likelihood")
            likelihood_function.save(save_file, identifier=landscape_name)

        time2 = time.time()
        run_stats["Make_Likelihoods_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['fit_landscapes']['run']:
        print("\nFitting Landscapes...\n")
        time1 = time.time()

        if config_dict['resolution_testing_options']['fit_landscapes']['overwrite_results_file']:
            all_fit_results = {}
        else:
            filename = os.path.join("GeneratedData", "RasterFits", "FitResults.json")
            try:
                with open(filename, "r") as fin:
                    all_fit_results = json.load(fin)
            except FileNotFoundError:
                all_fit_results = {}

        for resolution in test_resolutions:
            t1 = time.time()
            landscape_name = "ROI_" + str(resolution) + "Landscape"
            raster_header = raster_tools.RasterData.from_file(
                os.path.join("GeneratedData", landscape_name, "HostNumbers.txt")).header_vals

            kernel_names = config_dict['resolution_testing_options']['kernel_names']
            kernel_priors = [{} for _ in kernel_names]
            param_start = [{} for _ in kernel_names]
            for i, kernel in enumerate(kernel_names):
                params = config_dict['resolution_testing_options']['kernel_priors'][i].keys()
                for param in params:
                    kernel_priors[i][param] = config_dict['resolution_testing_options'][
                        'kernel_priors'][i][param]
                    param_start[i][param] = config_dict['resolution_testing_options'][
                        'kernel_init'][i][param]
                    if param == "scale":
                        kernel_priors[i][param] = np.array(kernel_priors[i][param]) / resolution
                        param_start[i][param] = np.array(param_start[i][param]) / resolution
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

            save_file = os.path.join("GeneratedData", "SimulationRuns",
                                     landscape_name+"_likelihood.npz")
            lik_loaded = raster_model_fitting.LikelihoodFunction.from_file(save_file)

            zipped_values = zip(kernel_names, kernel_generators, kernel_jac_generators,
                                kernel_priors, param_start)

            for name, gen, jac, prior, start in zipped_values:

                opt_params, fit_output = raster_model_fitting.fit_raster_MLE(
                    data_stub=os.path.join("GeneratedData", "SimulationRuns", "output"),
                    kernel_generator=gen, kernel_params=prior, param_start=start,
                    target_raster=raster_header, nsims=None, likelihood_func=lik_loaded,
                    kernel_jac=jac, raw_output=True
                )

                print("Complete. {0}, {1} kernel: {2}".format(landscape_name, name, opt_params))

                opt_params['Raw_Output'] = fit_output.__repr__()

                # Save results
                if landscape_name in all_fit_results:
                    all_fit_results[landscape_name][name] = opt_params
                else:
                    all_fit_results[landscape_name] = {
                        name: opt_params
                    }

                outfile = os.path.join(
                    "GeneratedData", "RasterFits", "FitResults.json")
                with open(outfile, "w") as f_out:
                    json.dump(all_fit_results, f_out, indent=4)

                if config_dict['resolution_testing_options']['fit_landscapes']['make_plots']:
                    plot_likelihood(landscape_name, name, gen, opt_params, lik_loaded, config_dict,
                                    prior)

            t2 = time.time()
            time_taken = timedelta(seconds=t2-t1)
            print("Landscape ROI_" + str(resolution) +
                  " Fitted. Time taken: {0}s".format(str(time_taken)))

        time2 = time.time()
        run_stats["Fit_Landscapes_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['test_fits']['run']:
        print("\nTesting Fits...\n")
        time1 = time.time()
        # Make and assess raster models
        all_budgets = []

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

        infile = os.path.join("GeneratedData", "RasterFits", "FitResults.json")
        with open(infile, "r") as fin:
            all_fit_results = json.load(fin)

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

                print(opt_params)

                coupling = np.ones((ncells_agg, ncells_agg))
                kernel = kernel_gen(**opt_params)

                control_rate = 0.005
                max_budget_rate = 500000

                for i in range(ncells_agg):
                    for j in range(ncells_agg):
                        dx = abs((i % dimensions_agg[1]) - (j % dimensions_agg[1]))
                        dy = abs(int(i/dimensions_agg[1]) - int(j/dimensions_agg[1]))
                        dist = np.sqrt(dx*dx + dy*dy)
                        coupling[i, j] = kernel(dist)

                times = np.linspace(0, 1000, 501)
                max_hosts = int(100 * np.power(resolution / full_resolution, 2))

                params = {
                    'dimensions': dimensions,
                    'inf_rate': 1.0,
                    'control_rate': control_rate,
                    'max_budget_rate': max_budget_rate,
                    'coupling': coupling,
                    'times': times,
                    'max_hosts': max_hosts,
                }

                host_file = os.path.join("GeneratedData", landscape_name, "HostDensity.txt")
                s_init_file = os.path.join("GeneratedData", landscape_name,
                                           "InitialConditions_Density_S.txt")
                i_init_file = os.path.join("GeneratedData", landscape_name,
                                           "InitialConditions_Density_I.txt")

                approx_model = raster_model.RasterModel(
                    params, host_density_file=host_file, initial_s_file=s_init_file,
                    initial_i_file=i_init_file)

                # Run policy (no control)
                def control_policy(t):
                    return [0]*ncells_agg

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

                print(no_control_metrics)

                tested_fit.save(os.path.join(
                    "GeneratedData", "RasterFits", landscape_name+"FitTest"))

                # Calculate quality metrics with control
                for budget in all_budgets:
                    # Run policy (with control, prioritised by N->S, E->W)
                    # Aggregate or divide data
                    # Calculate quality metric
                    pass

        time2 = time.time()
        run_stats["Fit_Landscapes_Time"] = str(timedelta(seconds=time2-time1))

    if config_dict['resolution_testing_options']['plot_fits']['run']:
        # Make plots assessing fit quality

        # Combine data
        metric_scales = ["Divided", "Normal", "Landscape"]
        all_data = {kernel: {metric: [] for metric in metric_scales} for kernel in kernel_names}
        for resolution in test_resolutions:
            nhosts = int(100 * np.power(resolution / full_resolution, 2))
            landscape_name = "ROI_" + str(resolution) + "Landscape"
            tested_fit = pickle.load(os.path.join(
                "GeneratedData", "RasterFits", landscape_name+"FitTest"))
            for kernel in kernel_names:
                all_data[kernel]["Divided"].append(
                    tested_fit.metric_data[kernel]["Divided"] / nhosts)
                all_data[kernel]["Normal"].append(
                    tested_fit.metric_data[kernel]["Normal"] / nhosts)
                all_data[kernel]["Landscape"].append(
                    tested_fit.metric_data[kernel]["Landscape"] / nhosts)

        # Make plots for each kernel
        for kernel in kernel_names:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for metric in metric_scales:
                ax.plot(test_resolutions, all_data[kernel][metric], label=metric)
            ax.legend()
            ax.set_title(kernel + " Kernel")
            ax.set_xlabel("Resolution / m")
            ax.set_ylabel("RMSE as Proportion of Cell")
            fig.savefig(os.path.join("Figures", "ResolutionTesting_" + kernel + ".png"))

        # Make plots for each metric
        for metric in metric_scales:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for kernel in kernel_names:
                ax.plot(test_resolutions, all_data[kernel][metric], label=kernel)
            ax.legend()
            ax.set_title(metric + " Metric")
            ax.set_xlabel("Resolution / m")
            ax.set_ylabel("RMSE as Proportion of Cell")
            fig.savefig(os.path.join("Figures", "ResolutionTesting_" + metric + ".png"))


    return run_stats


def plot_likelihood(landscape_name, name, gen, opt_params, lik_loaded, config_dict, prior):
    recalc_plot_vals = config_dict[
        'resolution_testing_options']['fit_landscapes']['recalculate_plot_values']

    all_params = {}
    param_names = []
    for param, value in opt_params.items():
        if param == "Raw_Output":
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
                    param_values = {param: all_meshed[k][i, j] for k, param in enumerate(param_names)}
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
                     metric="RMSE"):
    """Calculate fit quality metric for run_data, using base_data as simulation comparison."""

    run_data_div = {}

    run_data_total = np.zeros_like(run_data[0])
    run_data_total[:, 0] = run_data[0][:, 0]

    forward_map, back_map = raster_tools.aggregate_cells(
        base_raster_header, run_raster_header, generate_reverse=True, ignore_outside_target=True)
    n_cells = np.power(run_raster_header['cellsize']/base_raster_header['cellsize'], 2)

    for agg_cell, run in run_data.items():
        cells = back_map[agg_cell]
        for cell in cells:
            run_data_div[cell] = np.column_stack((
                run[:, 0],
                np.divide(run[:, 1], n_cells),
                np.divide(run[:, 2], n_cells)))

            run_data_total[:, 1] += np.divide(run_data[agg_cell][:, 1], n_cells)
            run_data_total[:, 2] += np.divide(run_data[agg_cell][:, 2], n_cells)

    all_run_data = {
        "Divided": run_data_div,
        "Normal": run_data,
        "Landscape": run_data_total
    }

    if metric == "RMSE":
        metric_vals, cell_data = calculate_rmse(tested_fit, all_run_data, run_raster_header)
    else:
        raise ValueError("Metric not recognised!")

    # tested_fit.add_kernel(kernel_name, all_run_data, cell_data, metric_vals)

    return metric_vals


def calculate_rmse(tested_fit, all_run_data, run_raster_header, verbose=True):
    """Calculate RMSE metrics at divided, normal and whole landscape scales."""

    run_data_div = all_run_data['Divided']
    run_data = all_run_data['Normal']
    run_data_landscape = all_run_data['Landscape']
    test_times = tested_fit.test_times

    nRuns = tested_fit.sim_data["Normal"].shape[0]

    nrows = run_raster_header['nrows']
    ncols = run_raster_header['ncols']
    nCells = nrows * ncols

    # Find correct time points in run_data
    all_times = run_data[0][:, 0]
    time_idxs = np.nonzero([x in test_times for x in all_times])[0]

    sse = {i: 0 for i in range(nCells)}

    # Aggregated rmse
    for i, dataset in enumerate(tested_fit.sim_data["Normal"]):
        for cell in range(nCells):
            # Find sum square error in each cell
            errors = [dataset[j, cell] - run_data[cell][idx, 2]
                      for j, idx in enumerate(time_idxs)]
            sse_tmp = np.sum(np.square(errors))
            sse[cell] += sse_tmp

        if verbose:
            print("Run {0} complete.".format(i))

    cell_rmses = [np.sqrt(sse[i] / (len(test_times)*nRuns)) for i in range(nCells)]
    cell_rmses = np.reshape(cell_rmses, (nrows, ncols))

    agg_rmse = np.sqrt(np.sum([sse[i] for i in range(nCells)]) / (len(time_idxs)*nCells*nRuns))

    # Divided rmse
    nCells_div = tested_fit.sim_data["Divided"][0].shape[1]
    nrows_div = tested_fit.base_raster.header_vals['nrows']
    ncols_div = tested_fit.base_raster.header_vals['ncols']

    sse = {i: 0 for i in range(nCells_div)}

    div_rmse = None
    cell_rmses_div = None
    for i, dataset in enumerate(tested_fit.sim_data["Divided"]):
        for cell in range(nCells_div):
            # Find sum square error in each cell
            errors = [dataset[j, cell] - run_data_div[cell][idx, 2]
                      for j, idx in enumerate(time_idxs)]
            sse_tmp = np.sum(np.square(errors))
            sse[cell] += sse_tmp

        if verbose:
            print("Run {0} complete.".format(i))

    cell_rmses_div = [np.sqrt(sse[i] / (len(test_times)*nRuns)) for i in range(nCells_div)]
    cell_rmses_div = np.reshape(cell_rmses_div, (nrows_div, ncols_div))

    div_rmse = np.sqrt(np.sum([sse[i] for i in range(nCells_div)]) / (
        len(time_idxs)*nCells_div*nRuns))

    # Landscape rmse
    errors = [[tested_fit.sim_data["Landscape"][i, j] - run_data_landscape[idx, 2]
               for j, idx in enumerate(time_idxs)] for i in range(nRuns)]
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

    return metric_data, cell_data
