""" Tools for summarising DPCs of simulation runs."""

from IPython import embed
import logging
import os
import h5py
import numpy as np
import pyproj
import raster_tools
from Scripts import MainOptions

def summarise_sims(target_header, region, sim_stub, landscape_name, states=None):
    """Generate DPC summaries for simulation runs."""

    if states is None:
        states = ['I']

    logging.info("Starting simulation DPC summaries for %s", landscape_name)

    llcorner, urcorner = region

    os.makedirs(os.path.join(sim_stub, "summaries"), exist_ok=True)

    n_cells = int(target_header['nrows'] * target_header['ncols'])
    # For 30 yrs, output every 2 weeks
    n_times = len(MainOptions.OPTIONS['times'])

    # Map projections
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    NAD83_Cali_Albers = pyproj.Proj("+init=EPSG:3310")

    llcorner_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *llcorner)
    urcorner_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *urcorner)

    n_sims = len([x for x in os.listdir(sim_stub) if x.startswith("rasters_")])
    sim_summary = [np.zeros((n_sims, n_cells, n_times)) for state in states]

    if os.path.isfile(os.path.join(sim_stub, "summaries", landscape_name + '.h5')):
        file_path = os.path.join(sim_stub, "summaries", landscape_name + ".h5")
        with h5py.File(file_path, 'r') as hf:
            for i, state in enumerate(states):
                summary_existing = hf['sim_summary_' + state]
                sim_summary[i][:len(summary_existing)] = summary_existing
                start = len(summary_existing)

    for sim in range(start, n_sims):

        for j, state in enumerate(states):
            # Output is every 2 weeks
            for i, week in enumerate(np.arange(0, 841, 2)):
                # if week > 0:
                #     week = str(week) + '.0'

                try:
                    filename = (sim_stub + "/rasters_{}/raster_{}_{}_{}.txt".format(
                        sim, sim, state, week))
                    raster = raster_tools.extract_raster(filename, llcorner_3310, urcorner_3310,
                                                         resolution=target_header['cellsize'])
                except FileNotFoundError:
                    # Here add logic for if epidemic has finished
                    # Simulation model now ouputs to end regardless
                    continue
                raster.array[raster.array == -9999] = 0
                raster.array *= (target_header['cellsize'] / 250)**2

                sim_summary[j][sim, :, i] = raster.array.flatten()

        logging.info("Read in data for simulation %d of %d", sim+1, n_sims)

    with h5py.File(os.path.join(sim_stub, "summaries", landscape_name + '.h5'), 'w') as hf:
        for j, state in enumerate(states):
            hf.create_dataset("sim_summary_" + state, data=sim_summary[j], compression='gzip')

    logging.info("Completed summary for %s", landscape_name)
