"""Tools for running optimisers, and running simulations with open-loop or buffer strategy."""

import json
import os
import shutil
import subprocess
import numpy as np
import raster_tools
from RasterModel import raster_model
from Scripts import MainOptions
import IndividualSimulator
from Scripts import summarise_sims

def run_optimiser(out_folder, obj_raster, scale_control_rate=True, landscape_name=None):
    """Run Ipopt optimiser, using 2500m resolution Cauchy kernel model.
        Objective raster must match landscape!"""

    if landscape_name is None:
        landscape_name = 'ROI_2500Landscape'
    kernel_name = 'Cauchy'

    # First get kernel parameters
    with open(os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"), "r") as f_in:
        fit_results = json.load(f_in)

    beta = fit_results[landscape_name][kernel_name]['beta']
    scale = fit_results[landscape_name][kernel_name]['scale']

    beta *= 100 * np.power(obj_raster.header_vals['cellsize']/250, 2)

    control_scaling = np.power(2500 / obj_raster.header_vals['cellsize'], 2)
    control_rate = MainOptions.OPTIONS['control_rate'] * control_scaling
    if scale_control_rate:
        file = os.path.join("GeneratedData", "ResolutionTesting", "scaling_results.json")
        with open(file, "r") as f_in:
            scaling_factor = json.load(f_in)['control_rate_factor']
        control_rate *= scaling_factor


    # Create folders and files
    os.makedirs(os.path.join("GeneratedData", out_folder), exist_ok=True)
    files_src = [
        'HostDensity.txt', 'InitialConditions_Density_S.txt', 'InitialConditions_Density_I.txt',
        'RMSMask.txt']
    files_dst = ['HostDensity_raster.txt', 'S0_raster.txt', 'I0_raster.txt', 'RMSMask.txt']
    for x, y in zip(files_src, files_dst):
        shutil.copyfile(
            src=os.path.join("GeneratedData", landscape_name, x),
            dst=os.path.join("GeneratedData", out_folder, y)
        )
    shutil.copyfile(
        src=os.path.join("..", "RasterModel", "ipopt.opt"),
        dst=os.path.join("GeneratedData", out_folder, "ipopt.opt")
    )

    obj_raster.to_file(os.path.join("GeneratedData", out_folder, "ObjWeights.txt"))

    # Create optimiser config file
    config_str = "method=1\nbeta={}\nscale={}\ncontrol_rate={}\nbudget=1\nfinal_time=30\n".format(
        beta, scale, control_rate)
    config_str += "n_segments=120\nmax_hosts=1\ncontrol_skip=3\ncontrol_start=12\n"
    config_str += "sus_file=RMSMask.txt\ninf_file=RMSMask.txt\nobj_file=ObjWeights.txt\n"

    with open(os.path.join("GeneratedData", out_folder, "OptimConfig.ini"), "w") as out_file:
        out_file.write(config_str)

    # Run Ipopt
    IPOPT_PATH = os.path.join("..", "..", "..", "RasterModel", "Ipopt", "RasterModel.exe")
    subprocess.run(
        [IPOPT_PATH, "OptimConfig.ini"], shell=True, cwd=os.path.join("GeneratedData", out_folder))

def optimise_non_spatial(out_folder, obj_raster, scale_control_rate=True, landscape_name=None):
    """Run Ipopt optimiser, using Cauchy kernel model for given landscape.
        Objective raster must match landscape!"""

    if landscape_name is None:
        landscape_name = 'ROI_2500Landscape'
    kernel_name = 'Cauchy'

    # First get kernel parameters
    with open(os.path.join("GeneratedData", "ResolutionTesting", "FitResults.json"), "r") as f_in:
        fit_results = json.load(f_in)

    beta = fit_results[landscape_name][kernel_name]['beta']
    scale = fit_results[landscape_name][kernel_name]['scale']

    beta *= 100 * np.power(obj_raster.header_vals['cellsize']/250, 2)

    control_scaling = np.power(2500 / obj_raster.header_vals['cellsize'], 2)
    control_rate = MainOptions.OPTIONS['control_rate'] * control_scaling
    if scale_control_rate:
        file = os.path.join("GeneratedData", "ResolutionTesting", "scaling_results.json")
        with open(file, "r") as f_in:
            scaling_factor = json.load(f_in)['control_rate_factor']
        control_rate *= scaling_factor

    # Create folders and files
    os.makedirs(os.path.join("GeneratedData", out_folder), exist_ok=True)
    files_src = [
        'HostDensity.txt', 'InitialConditions_Density_S.txt', 'InitialConditions_Density_I.txt',
        'RMSMask.txt']
    files_dst = ['HostDensity_raster.txt', 'S0_raster.txt', 'I0_raster.txt', 'RMSMask.txt']
    for x, y in zip(files_src, files_dst):
        shutil.copyfile(
            src=os.path.join("GeneratedData", landscape_name, x),
            dst=os.path.join("GeneratedData", out_folder, y)
        )
    shutil.copyfile(
        src=os.path.join("..", "RasterModel", "ipopt.opt"),
        dst=os.path.join("GeneratedData", out_folder, "ipopt.opt")
    )

    obj_raster.to_file(os.path.join("GeneratedData", out_folder, "ObjWeights.txt"))

    # Create optimiser config file
    config_str = "method=1\nbeta={}\nscale={}\ncontrol_rate={}\nbudget=1\nfinal_time=30\n".format(
        beta, scale, control_rate)
    config_str += "n_segments=120\nmax_hosts=1\ncontrol_skip=4\nnon_spatial=1\ncontrol_start=12\n"
    config_str += "sus_file=RMSMask.txt\ninf_file=RMSMask.txt\nobj_file=ObjWeights.txt\n"

    with open(os.path.join("GeneratedData", out_folder, "OptimConfig.ini"), "w") as out_file:
        out_file.write(config_str)

    # Run Ipopt
    IPOPT_PATH = os.path.join("..", "..", "..", "RasterModel", "Ipopt", "RasterModel.exe")
    subprocess.run(
        [IPOPT_PATH, "OptimConfig.ini"], shell=True, cwd=os.path.join("GeneratedData", out_folder))

def run_buffer(out_folder, iterations, start=0):
    """Run 100m buffer simulations."""

    params = IndividualSimulator.code.config.read_config_file(
        filename=os.path.join("InputData", "REDW_config.ini"))

    for i in range(start, start+iterations):

        params['FinalTime'] = 28 * 30
        params['OutputFileStub'] = ('{}/output'.format(out_folder))
        params['RasterFileStub'] = ('{}/rasters_{}/raster'.format(out_folder, i))
        params['HostPosFile'] = 'GeneratedData/EROI_withC/HostNumbers.txt'
        params['InitCondFile'] = 'GeneratedData/EROI_withC/InitialConditions_Numbers'
        os.makedirs('{}/rasters_{}'.format(out_folder, i), exist_ok=True)

        params['SusceptibilityFile'] = 'GeneratedData/EROI_withC/RMSMask.txt'
        params['InfectiousnessFile'] = 'GeneratedData/EROI_withC/RMSMask.txt'
        params['RasterOutputFreq'] = 2
        params['OutputEventData'] = False
        params['OutputHostData'] = False
        params['OutputFiles'] = False

        params['InterventionScripts'] = 'Scripts.BufferIntervention'
        params['InterventionUpdateFrequencies'] = '28'
        params['UpdateOnAllEvents'] = True
        intervention_options = '98,100,0.7,124,GeneratedData/ROI_2500Landscape/HostNumbers.txt,'
        intervention_options += 'GeneratedData/EROI_withC/HostNumbers.txt'
        params['InterventionOptions'] = intervention_options

        params['RateStructure-Advance'] = 'ratetree'

        params['Model'] = 'SCIR'
        params['CAdvRate'] = 1000000 # No need for cryptic period - set to very large C->I rate

        run_sim = IndividualSimulator.Simulator(params)
        run_sim.setup()

        # Randomly position hosts within cells
        for host in run_sim.params['init_hosts']:
            host.xpos += 250 * (np.random.rand() - 0.5)
            host.ypos += 250 * (np.random.rand() - 0.5)

        run_sim.initialise()
        run_sim.run_epidemic(iteration=i)

    # Summarise sims at aggregated scale
    raster_header = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "ROI_2500Landscape", "HostNumbers.txt")).header_vals
    landscape_name = 'ROI_2500Landscape'
    summarise_sims.summarise_sims(
        target_header=raster_header, region=MainOptions.OPTIONS['roi'],
        sim_stub=out_folder, landscape_name=landscape_name, states=['S', 'I', 'C', 'Culled'])

    # Summarise sims at full resolution
    raster_header = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "ROI_250Landscape", "HostNumbers.txt")).header_vals
    landscape_name = 'ROI_250Landscape'
    summarise_sims.summarise_sims(
        target_header=raster_header, region=MainOptions.OPTIONS['roi'],
        sim_stub=out_folder, landscape_name=landscape_name, states=['S', 'I', 'C', 'Culled'])

    # Summarise sims at full resolution (EROI)
    raster_header = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "EROI", "HostNumbers.txt")).header_vals
    landscape_name = 'EROI'
    summarise_sims.summarise_sims(
        target_header=raster_header, region=MainOptions.OPTIONS['eroi'],
        sim_stub=out_folder, landscape_name=landscape_name, states=['S', 'I', 'C', 'Culled'])


def run_open_loop(optimiser_output_folder, out_folder, iterations, start=0, landscape_name=None):
    """Run open-loop simulations using raster optimiser output as control."""

    if landscape_name is None:
        landscape_name = 'ROI_2500Landscape'

    params = IndividualSimulator.code.config.read_config_file(
        filename=os.path.join("InputData", "REDW_config.ini"))

    host_raster = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", landscape_name, "HostNumbers.txt"))

    control_scaling = np.power(2500 / host_raster.header_vals['cellsize'], 2)
    control_rate = MainOptions.OPTIONS['control_rate'] * control_scaling / 28

    for i in range(start, start+iterations):

        params['FinalTime'] = 28 * 30
        params['OutputFileStub'] = ('{}/output'.format(out_folder))
        params['RasterFileStub'] = ('{}/rasters_{}/raster'.format(out_folder, i))
        params['HostPosFile'] = 'GeneratedData/EROI/HostNumbers.txt'
        params['InitCondFile'] = 'GeneratedData/EROI/InitialConditions_Numbers'
        os.makedirs('{}/rasters_{}'.format(out_folder, i), exist_ok=True)

        params['SusceptibilityFile'] = 'GeneratedData/EROI/RMSMask.txt'
        params['InfectiousnessFile'] = 'GeneratedData/EROI/RMSMask.txt'
        params['RasterStatesOutput'] = ['S', 'I', 'Culled']
        params['RasterOutputFreq'] = 2
        params['OutputEventData'] = False
        params['OutputHostData'] = False
        params['OutputFiles'] = False

        params['InterventionScripts'] = 'Scripts.OLIntervention'
        params['InterventionUpdateFrequencies'] = '1'
        params['UpdateOnAllEvents'] = True
        intervention_options = 'GeneratedData/{}/HostNumbers.txt,'.format(landscape_name)
        intervention_options += 'GeneratedData/EROI/HostNumbers.txt,'
        intervention_options += '{},{}'.format(control_rate, os.path.join(
            "GeneratedData", optimiser_output_folder, 'output'))
        params['InterventionOptions'] = intervention_options

        IndividualSimulator.run_epidemics(params, iteration_start=i)

    # Summarise sims at aggregated scale
    raster_header = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "ROI_2500Landscape", "HostNumbers.txt")).header_vals
    landscape_name = 'ROI_2500Landscape'
    summarise_sims.summarise_sims(
        target_header=raster_header, region=MainOptions.OPTIONS['roi'],
        sim_stub=out_folder, landscape_name=landscape_name, states=['S', 'I', 'Culled'])

    # Summarise sims at full resolution (ROI)
    raster_header = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "ROI_250Landscape", "HostNumbers.txt")).header_vals
    landscape_name = 'ROI_250Landscape'
    summarise_sims.summarise_sims(
        target_header=raster_header, region=MainOptions.OPTIONS['roi'],
        sim_stub=out_folder, landscape_name=landscape_name, states=['S', 'I', 'Culled'])

    # Summarise sims at full resolution (EROI)
    raster_header = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", "EROI", "HostNumbers.txt")).header_vals
    landscape_name = 'EROI'
    summarise_sims.summarise_sims(
        target_header=raster_header, region=MainOptions.OPTIONS['eroi'],
        sim_stub=out_folder, landscape_name=landscape_name, states=['S', 'I', 'Culled'])

def generate_obj_raster(landscape_name, outside_np_factor=0.0):
    """Generate objective raster using NP mask."""

    obj_raster = raster_tools.RasterData.from_file(os.path.join(
        "GeneratedData", landscape_name, "NPMask.txt"))

    outside_np_array = (1.0 - obj_raster.array) * outside_np_factor
    obj_raster.array += outside_np_array

    return obj_raster
