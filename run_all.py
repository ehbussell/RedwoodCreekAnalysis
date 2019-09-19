"""Run full analysis for Redwood Creek project."""

import logging
import os
import argparse
import pickle
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt

import IndividualSimulator
import raster_tools
from Scripts import generate_landscapes
from Scripts import kernels
from Scripts import resolution_testing
from Scripts import generate_kernel
from Scripts import MainOptions
from Scripts import sod_sim_funcs
from Scripts import optimise_control
from RasterModel import raster_model_fitting
from RasterModel import raster_model

def run_main_analysis(sim_stub, n_sims=10, sim_start=0, append=False):
    """Run analysis."""

    if not append:

        # # Kernel generation stage
        file_name = os.path.join("GeneratedData", "Kernel_Raster_250.txt")
        generate_kernel.generate_kernel(file_name)

        # Create EROI landscape
        logging.info("Creating EROI landscape.")
        generate_landscapes.generate_landscape(MainOptions.OPTIONS['eroi'], 250, 'EROI')

        # Create EROI landscape with Cryptic hosts
        os.makedirs("GeneratedData/EROI_withC", exist_ok=True)
        files_src = [
            'HostDensity.txt', 'HostNumbers.txt',
            'InitialConditions_Density_S.txt', 'InitialConditions_Density_I.txt',
            'InitialConditions_Density_R.txt', 'InitialConditions_Density_R.txt',
            'InitialConditions_Numbers_S.txt', 'InitialConditions_Numbers_I.txt',
            'InitialConditions_Numbers_R.txt', 'InitialConditions_Numbers_R.txt',
            'RMSMask.txt', 'NPMask.txt']
        files_dst = [
            'HostDensity.txt', 'HostNumbers.txt',
            'InitialConditions_Density_S.txt', 'InitialConditions_Density_C.txt',
            'InitialConditions_Density_I.txt', 'InitialConditions_Density_R.txt',
            'InitialConditions_Numbers_S.txt', 'InitialConditions_Numbers_C.txt',
            'InitialConditions_Numbers_I.txt', 'InitialConditions_Numbers_R.txt',
            'RMSMask.txt', 'NPMask.txt']
        for x, y in zip(files_src, files_dst):
            shutil.copyfile(
                src=os.path.join("GeneratedData", 'EROI', x),
                dst=os.path.join("GeneratedData", 'EROI_withC', y)
            )

        # Run 100 simulations under no control
        logging.info("Running simulations under no control.")
        for i in range(100):
            sod_sim_funcs.run_avg_sim(os.path.join(sim_stub, 'EROI'), 30, iteration=i)

        # Run resolution testing (no control)
        logging.info("Starting resolution testing (no control).")
        resolution_testing.run_no_control(os.path.join(sim_stub, 'EROI'))

        # Optimise non-spatial strategy using 2500m Cauchy kernel
        logging.info("Starting non-spatial control optimisation.")
        objective_raster = optimise_control.generate_obj_raster("ROI_2500Landscape", 1.0)
        optimise_control.optimise_non_spatial(
            'Total_non_spatial', objective_raster, scale_control_rate=False,
            landscape_name='ROI_2500Landscape')

        # Run simulations using non-spatial strategy
        optimise_control.run_open_loop(
            'Total_non_spatial', os.path.join(sim_stub, 'Total_non_spatial'), 100)

        # Resolution testing under non-spatial control strategy
        logging.info("Starting resolution testing (with control).")
        resolution_testing.run_with_control(
            os.path.join(sim_stub, 'Total_non_spatial'),
            os.path.join('GeneratedData', 'Total_non_spatial', 'output'))

        logging.info("Resolution testing complete")

        # Run optimiser for 'Total' objective at 2.5km resolution
        objective_raster = optimise_control.generate_obj_raster("ROI_2500Landscape", 1.0)
        optimise_control.run_optimiser("Total_2500", objective_raster, scale_control_rate=False,
                                       landscape_name='ROI_2500Landscape')

        # Run optimiser for 'NP' objective at 2.5km resolution
        objective_raster = optimise_control.generate_obj_raster("ROI_2500Landscape", 0.0)
        optimise_control.run_optimiser("NP_2500", objective_raster, scale_control_rate=False,
                                       landscape_name='ROI_2500Landscape')

        # Run optimiser for 'Mixed' objective at 2.5km resolution
        objective_raster = optimise_control.generate_obj_raster("ROI_2500Landscape", 0.5)
        optimise_control.run_optimiser("Mixed_2500", objective_raster, scale_control_rate=False,
                                       landscape_name='ROI_2500Landscape')

        # Run optimiser for 'Total' objective at 5km resolution
        objective_raster = optimise_control.generate_obj_raster("ROI_5000Landscape", 1.0)
        optimise_control.run_optimiser("Total_5000", objective_raster, scale_control_rate=False,
                                       landscape_name='ROI_5000Landscape')

        # Run optimiser for 'NP' objective at 5km resolution
        objective_raster = optimise_control.generate_obj_raster("ROI_5000Landscape", 0.0)
        optimise_control.run_optimiser("NP_5000", objective_raster, scale_control_rate=False,
                                       landscape_name='ROI_5000Landscape')

        # Run optimiser for 'Mixed' objective at 5km resolution
        objective_raster = optimise_control.generate_obj_raster("ROI_5000Landscape", 0.5)
        optimise_control.run_optimiser("Mixed_5000", objective_raster, scale_control_rate=False,
                                       landscape_name='ROI_5000Landscape')

    # Run controlled simulations
    logging.info("Starting simulations")
    control_strats = ['Total_2500', 'NP_2500', 'Mixed_2500', 'Total_5000', 'NP_5000', 'Mixed_5000']
    landscape_names = ['ROI_2500Landscape', 'ROI_2500Landscape', 'ROI_2500Landscape',
                       'ROI_5000Landscape', 'ROI_5000Landscape', 'ROI_5000Landscape']
    for strat, landscape in zip(control_strats, landscape_names):
        logging.info("Starting simulations for strategy: %s", strat)
        out_folder = os.path.join(sim_stub, strat)
        optimise_control.run_open_loop(strat, out_folder, n_sims, start=sim_start,
                                       landscape_name=landscape)

    logging.info("Starting simulations for strategy: 100mBuffer")
    out_folder = os.path.join(sim_stub, '100mBuffer')
    optimise_control.run_buffer(out_folder, n_sims, start=sim_start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sim_stub", help="Stub to simulation output folders.")
    parser.add_argument("-a", "--append", action="store_true", help="Append to existing simulation"
                        " results. No resolution testing or optimisations carried out.")
    parser.add_argument("-n", "--n_sims", default=10, type=int,
                        help="Number of simulations to run using OL strategies")
    parser.add_argument("-s", "--sim_start", default=0, type=int,
                        help="Iteration to start simulations from.")


    args = parser.parse_args()

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join("GeneratedData", 'analysis.log'))
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

    run_main_analysis(args.sim_stub, args.n_sims, args.sim_start, args.append)
