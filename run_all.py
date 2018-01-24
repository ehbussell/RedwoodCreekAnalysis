"""Run full analysis for Redwood Creek project."""

import os
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

import pymc3 as pm

import IndividualSimulator
import raster_tools
from Scripts import generate_landscapes
from Scripts import kernels
from Scripts import resolution_testing
from RasterModel import raster_model_fitting
from RasterModel import raster_model

def main(stages=None):
    """Run analysis."""

    if stages is None:
        stages = ["all"]

    roi = ((-124.1, 40.99), (-123.8, 41.2))
    eroi = ((-124.2, 40.89), (-123.7, 41.3))

    # roi_name = "ROI_Landscape"
    roi_name = "LowResLandscape"
    eroi_name = "EROI_Landscape"

    full_resolution = 250
    reduced_resolution = 2500

    # Landscape generation stage
    if "all" in stages or "genLandscapes" in stages:
        options = {'map_highlight_region': roi, 'map_detail': "f"}
        generate_landscapes.generate_landscape(eroi, full_resolution, eroi_name, options)
        options['map_highlight_region'] = None
        generate_landscapes.generate_landscape(roi, reduced_resolution, roi_name, options)

    # Simulation running stage
    if "all" in stages or "runSims" in stages:
        run_data = IndividualSimulator.main(os.path.join("InputData", "REDW_config.ini"))

    # Likelihood generation stage
    if "all" in stages or "genLikelihood" in stages:
        raster_header = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", roi_name, "HostNumbers.txt")).header_vals
        
        likelihood_function = raster_model_fitting.precompute_loglik(
                data_stub=os.path.join("GeneratedData", "SimulationRuns", "output"),
                nsims=None, raster_header=raster_header, end_time=1000, ignore_outside_raster=True,
                precompute_level="full")

        save_file = os.path.join("GeneratedData", "SimulationRuns", roi_name+"_likelihood")
        likelihood_function.save(save_file, identifier=roi_name)

    # Fit all kernels for raster models
    if "all" in stages or "fitKernels" in stages:
        kernel_names = ["Exponential"]
        kernel_generators = [kernels.make_exponential_kernel]
        kernel_priors = [[("Beta", (0, 0.01)), ("Scale", (0, 2))]]

        raster_header = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", roi_name, "HostNumbers.txt")).header_vals

        save_file = os.path.join("GeneratedData", "SimulationRuns", roi_name+"_likelihood.npz")
        lik_loaded = raster_model_fitting.LikelihoodFunction.from_file(save_file)

        for name, gen, prior in zip(kernel_names, kernel_generators, kernel_priors):
            trace = raster_model_fitting.fit_raster_MCMC(
                data_stub=os.path.join("GeneratedData", "SimulationRuns", "output"),
                kernel_generator=gen, kernel_params=prior, target_raster=raster_header,
                nsims=None, mcmc_params={'iters':5000},
                output_stub=os.path.join("GeneratedData", "RasterFits", name+"RasterFit"),
                likelihood_func=lik_loaded
            )

            fig, axs = plt.subplots(2, 2)
            pm.traceplot(trace, ax=axs)
            fig.savefig(os.path.join("Figures", roi_name+name+"MCMCTrace.png"))

            outfile = os.path.join("GeneratedData", "RasterFits", roi_name+name+"trace")
            np.savez_compressed(outfile, Beta=trace['Beta'], Scale=trace['Scale'])

    # Create and optimise raster model
    if "all" in stages or "optimise" in stages:
        savefile = os.path.join("GeneratedData", "RasterFits", roi_name+"Exponentialtrace.npz")
        trace = np.load(savefile)

        beta = np.mean(trace['Beta'])
        scale = np.mean(trace['Scale'])

        print(beta, scale)

        raster_header = raster_tools.RasterData.from_file(
            os.path.join("GeneratedData", roi_name, "HostDensity.txt")).header_vals

        dimensions = (raster_header['nrows'], raster_header['ncols'])
        ncells = np.prod(dimensions)
        coupling = np.full((ncells, ncells), 1.0)

        control_rate = 0.005
        max_budget_rate = 500000

        for i in range(ncells):
            for j in range(ncells):
                dx = abs((i % dimensions[1]) - (j % dimensions[1]))
                dy = abs(int(i/dimensions[1]) - int(j/dimensions[1]))
                dist = np.sqrt(dx*dx + dy*dy)
                coupling[i, j] = np.exp(-dist/scale) / (2 * np.pi * scale * scale)

        times = np.linspace(0, 1000, 2001)
        max_hosts = int(100 * np.power(reduced_resolution / full_resolution, 2))

        params = {
            'dimensions': dimensions,
            'inf_rate': beta,
            'control_rate': control_rate,
            'max_budget_rate': max_budget_rate,
            'coupling': coupling,
            'times': times,
            'max_hosts': max_hosts,
        }

        host_file = os.path.join("GeneratedData", roi_name, "HostDensity.txt")
        s_init_file = os.path.join("GeneratedData", roi_name, "InitialConditions_Density_S.txt")
        i_init_file = os.path.join("GeneratedData", roi_name, "InitialConditions_Density_I.txt")

        approx_model = raster_model.RasterModel(
            params, host_density_file=host_file, initial_s_file=s_init_file,
            initial_i_file=i_init_file)

        def control_policy(t):
            return [0]*ncells

        results = approx_model.run_scheme(control_policy)
        results.plot()

        ipopt_options = {
            "max_iter": 500,
            "tol": 1.0e-6,
            "linear_solver": "ma97",
            "check_derivatives_for_naninf": "yes",
            "ma86_scaling": "none",
            "ma86_order": "metis",
            "mu_strategy": "adaptive",
            "hessian_constant": "no"}

        results2 = approx_model.optimise_Ipopt(ipopt_options, method="midpoint")
        results2.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-s", "--stages", help="Which stages of the analysis to run. Default: 'all'"
                        ". Space separated list of stages chosen from: genLandscapes, runSims, "
                        "genLikelihood, fitKernels, optimise",
                        nargs="+", default=["all"])
    parser.add_argument("-r", "--resolutionTesting", help="Flag to run all resolution tests. "
                        "Default is not to run.", action="store_true")
    parser.add_argument("-u", "--resolutionStages", help="Which stages of the resolution tests to "
                        "run. Default: 'all'. Space separated list of stages chosen from: "
                        "fitLandscapes",
                        nargs="+", default=["all"])
    parser.add_argument("-t", "--testResolutions", help="Which resolutions to test in resolution "
                        "testing. Default: 250, 2500. Space separated list of resolutions.",
                        nargs="+", default=None, type=int)
    args = parser.parse_args()

    if args.resolutionTesting:
        resolution_testing.main(args.resolutionStages, args.testResolutions)
    else:
        main(args.stages)
