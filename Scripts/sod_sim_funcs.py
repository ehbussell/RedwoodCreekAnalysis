"""Functions for running SOD simulations - including full weather data or averaged"""

import pdb
from IPython import embed
import inspect
import numpy as np
import os
import IndividualSimulator
import raster_tools
import matplotlib.pyplot as plt
import time

def run_avg_sim(sim_stub, num_years, infect_cell=None, iteration=0):
    """Run simulation for num_years, using constant averaged forest mask and weather data."""

    analysis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    params = IndividualSimulator.code.config.read_config_file(
        filename=os.path.join(analysis_path, "InputData", "REDW_config.ini"))

    params['FinalTime'] = 28 * num_years
    params['OutputFileStub'] = os.path.join(sim_stub, 'output')
    params['RasterFileStub'] = os.path.join(sim_stub, 'rasters_{}/raster'.format(iteration))
    params['HostPosFile'] = 'GeneratedData/EROI/HostNumbers.txt'
    params['InitCondFile'] = 'GeneratedData/EROI/InitialConditions_Numbers'
    os.makedirs(os.path.join(sim_stub, 'rasters_{}'.format(iteration)), exist_ok=True)

    params['SusceptibilityFile'] = 'GeneratedData/EROI/RMSMask.txt'
    params['InfectiousnessFile'] = 'GeneratedData/EROI/RMSMask.txt'
    params['RasterStatesOutput'] = 'SI'
    params['RasterOutputFreq'] = 2
    params['OutputEventData'] = False
    params['OutputHostData'] = False
    params['OutputFiles'] = False

    IndividualSimulator.run_epidemics(params, iteration_start=iteration)

def run_full_sim(sim_stub, num_years, landscape_name="ROI_Landscape", infect_cell=None, iteration=0):
    """Run full simulation for num_years, using variable forest mask and weather data.

    Note weather data will start from 1990.
    """

    analysis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    params = IndividualSimulator.code.config.read_config_file(
        filename=os.path.join(analysis_path, "InputData", "REDW_config.ini"))

    forest_mask = raster_tools.RasterData.from_file(os.path.join(
        analysis_path, "GeneratedData", landscape_name, "ForestMask.txt"))
    forest_mask.array[forest_mask.array == -9999] = np.nan

    header = forest_mask.header_vals

    susceptibility_raster = raster_tools.RasterData(
        shape=(header['nrows'], header['ncols']),
        llcorner=(header['xllcorner'], header['yllcorner']),
        cellsize=header['cellsize'],
        NODATA_value=header['NODATA_value'],
        array=np.ones((header['nrows'], header['ncols']))
    )

    params['FinalTime'] = 1
    params['OutputFileStub'] = os.path.join(sim_stub, 'output')
    params['RasterFileStub'] = os.path.join(sim_stub, 'rasters_{}/raster'.format(iteration))
    os.makedirs(
        os.path.join(sim_stub, 'rasters_{}'.format(iteration)),
        exist_ok=True)

    weather_stub = os.path.join(analysis_path, "InputData", "weather", "gis_m_c_")
    llcorner = (header['xllcorner'], header['yllcorner'])
    urcorner = (header['xllcorner'] + (header['ncols'] - 0.5) * header['cellsize'],
                header['yllcorner'] + (header['nrows'] - 0.5) * header['cellsize'])

    run_sim = IndividualSimulator.Simulator(params)
    run_sim.setup()
    run_sim.initialise()

    IndividualSimulator.code.outputdata.output_raster_data(
        run_sim, time=0, iteration=iteration, states=['S', 'I'])
    overall_week = 1

    # Choose random cell to infect
    if infect_cell is not None:
        run_sim.params['init_cells'][infect_cell].states['I'] = 1
        run_sim.params['init_cells'][infect_cell].states['S'] -= 1
        print('Infected cell {}'.format(infect_cell))

    for year in range(1990, 1990+num_years):
        for week in range(1, 29):
            # Extract correct region of weather file
            weather_raster = raster_tools.extract_raster(
                weather_stub + str(year) + "_" + str(week) + ".txt", llcorner, urcorner)

            susceptibility_raster.array = weather_raster.array

            # Set mixed evergreen forest type inactive for first 6 weeks
            if week < 7:
                susceptibility_raster.array[forest_mask.array == 2] = 0

            IndividualSimulator.code.hosts.read_sus_inf_files(
                run_sim.params['init_cells'], run_sim.params['header'],
                susceptibility_raster, susceptibility_raster,
                sim_type=run_sim.params['SimulationType'])
            
            # Calculate intial rates
            init_inf_rates = np.zeros(run_sim.params['ncells'])
            if run_sim.params['VirtualSporulationStart'] is not None:
                init_spore_rates = np.zeros(run_sim.params['ncells'])
            init_adv_rates = np.zeros(run_sim.params['nhosts'])

            for cell in run_sim.params['init_cells']:
                for host in cell.hosts:
                    current_state = host.state
                    if current_state in "ECDI":
                        init_adv_rates[host.host_id] = run_sim.params[current_state + 'AdvRate']
                if (cell.states["C"] + cell.states["I"]) > 0:
                    for cell2_rel_pos in run_sim.params['coupled_positions']:
                        cell2_pos = tuple(item1 + item2 for item1, item2
                                          in zip(cell.cell_position, cell2_rel_pos))
                        cell2_id = run_sim.params['cell_map'].get(cell2_pos, None)
                        if cell2_id is None:
                            continue
                        cell2 = run_sim.params['init_cells'][cell2_id]
                        init_inf_rates[cell2_id] += (
                            cell2.susceptibility * cell2.states["S"] *
                            (cell.states["C"] + cell.states["I"]) * cell.infectiousness *
                            run_sim.event_handler.kernel(cell2_rel_pos) /
                            run_sim.params['MaxHosts'])

                    if run_sim.params['VirtualSporulationStart'] is not None:
                        init_spore_rates[cell.cell_id] = (
                            cell.states["C"] + cell.states["I"]) * cell.infectiousness

            run_sim.params['init_inf_rates'] = init_inf_rates
            if run_sim.params['VirtualSporulationStart'] is not None:
                run_sim.params['init_spore_rates'] = init_spore_rates
            run_sim.params['init_adv_rates'] = init_adv_rates

            run_sim.initialise(silent=True)
            hosts, cells, _ = run_sim.run_epidemic(silent=True)

            # Set final state as new initial conditions
            run_sim.params['init_hosts'] = hosts
            run_sim.params['init_cells'] = cells

            # Output rasters for this week
            IndividualSimulator.code.outputdata.output_raster_data(
                run_sim, time=overall_week, iteration=iteration, states=['S', 'I'])
            overall_week += 1


        cells_infected = np.sum([1 for x in cells if x.states["I"] > 0])
        hosts_infected = np.sum([(x.states["C"] + x.states["I"]) for x in cells])
        print("Year {} done. {} cells infected, {} hosts infected".format(
            year, cells_infected, hosts_infected))

    return cells_infected, hosts_infected

def run_weather_avg_sim(sim_stub, num_years, landscape_name="ROI_Landscape", infect_cell=None, iteration=0):
    """Run full simulation for num_years, using variable forest mask and averaged weather data.

    Note weather data will start from 1990.
    """

    analysis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    params = IndividualSimulator.code.config.read_config_file(
        filename=os.path.join(analysis_path, "InputData", "REDW_config.ini"))

    forest_mask = raster_tools.RasterData.from_file(os.path.join(
        analysis_path, "GeneratedData", landscape_name, "ForestMask.txt"))
    forest_mask.array[forest_mask.array == -9999] = np.nan

    weather_raster = raster_tools.RasterData.from_file(os.path.join(
        analysis_path, "GeneratedData", landscape_name, "RMSWeather.txt"))
    weather_raster.array[weather_raster.array == -9999] = np.nan

    header = forest_mask.header_vals

    susceptibility_raster = raster_tools.RasterData(
        shape=(header['nrows'], header['ncols']),
        llcorner=(header['xllcorner'], header['yllcorner']),
        cellsize=header['cellsize'],
        NODATA_value=header['NODATA_value'],
        array=np.ones((header['nrows'], header['ncols']))
    )

    params['FinalTime'] = 1
    params['OutputFileStub'] = os.path.join(sim_stub, 'output')
    params['RasterFileStub'] = os.path.join(sim_stub, 'rasters_{}/raster'.format(iteration))
    os.makedirs(
        os.path.join(sim_stub, 'rasters_{}/raster'.format(iteration)),
        exist_ok=True)
    params['RasterStatesOutput'] = 'SI'
    params['RasterOutputFreq'] = 0

    run_sim = IndividualSimulator.Simulator(params)
    run_sim.setup()
    run_sim.initialise()

    IndividualSimulator.code.outputdata.output_raster_data(
        run_sim, time=0, iteration=iteration, states=['S', 'I'])
    overall_week = 1

    # Choose cell to infect
    if infect_cell is not None:
        run_sim.params['init_cells'][infect_cell].states['I'] = 1
        run_sim.params['init_cells'][infect_cell].states['S'] -= 1
        print('Infected cell {}'.format(infect_cell))

    for year in range(1990, 1990+num_years):
        for week in range(1, 29):

            susceptibility_raster.array = weather_raster.array

            # Set mixed evergreen forest type inactive for first 6 weeks
            if week < 7:
                susceptibility_raster.array[forest_mask.array == 2] = 0

            IndividualSimulator.code.hosts.read_sus_inf_files(
                run_sim.params['init_cells'], run_sim.params['header'],
                susceptibility_raster, susceptibility_raster,
                sim_type=run_sim.params['SimulationType'])

            # Calculate intial rates
            init_inf_rates = np.zeros(run_sim.params['ncells'])
            if run_sim.params['VirtualSporulationStart'] is not None:
                init_spore_rates = np.zeros(run_sim.params['ncells'])
            init_adv_rates = np.zeros(run_sim.params['nhosts'])

            for cell in run_sim.params['init_cells']:
                for host in cell.hosts:
                    current_state = host.state
                    if current_state in "ECDI":
                        init_adv_rates[host.host_id] = run_sim.params[current_state + 'AdvRate']
                if (cell.states["C"] + cell.states["I"]) > 0:
                    for cell2_rel_pos in run_sim.params['coupled_positions']:
                        cell2_pos = tuple(item1 + item2 for item1, item2
                                          in zip(cell.cell_position, cell2_rel_pos))
                        cell2_id = run_sim.params['cell_map'].get(cell2_pos, None)
                        if cell2_id is None:
                            continue
                        cell2 = run_sim.params['init_cells'][cell2_id]
                        init_inf_rates[cell2_id] += (
                            cell2.susceptibility * cell2.states["S"] *
                            (cell.states["C"] + cell.states["I"]) * cell.infectiousness *
                            run_sim.event_handler.kernel(cell2_rel_pos) /
                            run_sim.params['MaxHosts'])

                    if run_sim.params['VirtualSporulationStart'] is not None:
                        init_spore_rates[cell.cell_id] = (
                            cell.states["C"] + cell.states["I"]) * cell.infectiousness

            run_sim.params['init_inf_rates'] = init_inf_rates
            if run_sim.params['VirtualSporulationStart'] is not None:
                run_sim.params['init_spore_rates'] = init_spore_rates
            run_sim.params['init_adv_rates'] = init_adv_rates

            run_sim.initialise(silent=True)
            hosts, cells, _ = run_sim.run_epidemic(silent=True)

            # Set final state as new initial conditions
            run_sim.params['init_hosts'] = hosts
            run_sim.params['init_cells'] = cells

            # Output rasters for this week
            IndividualSimulator.code.outputdata.output_raster_data(
                run_sim, time=overall_week, iteration=iteration, states=['S', 'I'])
            overall_week += 1


        cells_infected = np.sum([1 for x in cells if x.states["I"] > 0])
        hosts_infected = np.sum([(x.states["C"] + x.states["I"]) for x in cells])
        print("Year {} done. {} cells infected, {} hosts infected".format(
            year, cells_infected, hosts_infected))

    return cells_infected, hosts_infected
