"""Intervention class script for simulator implementing buffer zone culling."""

from IPython import embed
import time as mtime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import raster_tools

class Intervention:
    """Buffer intervention"""

    def __init__(self, update_freq, all_hosts, all_cells, options):

        start_time, buffer, detection_prob, ncols, control_raster_file, sim_raster_file = options
        control_raster_header = raster_tools.RasterData.from_file(control_raster_file).header_vals
        sim_raster_header = raster_tools.RasterData.from_file(sim_raster_file).header_vals

        self.sim_to_control_cell_map = raster_tools.aggregate_cells(
            sim_raster_header, control_raster_header, generate_reverse=False,
            ignore_outside_target=True)

        self.start_time = float(start_time)
        self.buffer = float(buffer)
        self.detection_prob = float(detection_prob)
        self.ncols = int(ncols)

        # Set required properties: update frequency, intervention type, & rate structure size
        self.update_freq = update_freq
        self.type = "BUFFER"

        self.infected_hosts = set()

        # Perform initial setup
        for host in all_hosts:
            if host.cell_id in self.sim_to_control_cell_map:
                if host.state == 'I':
                    self.infected_hosts.add(host.host_id)

    def update(self, all_hosts, time, all_cells=None, after_event=None, get_rate_fn=None,
               initial=False):

        if initial:
            raise RuntimeError("Initial update called on non-CONTINUOUS intervention")

        # Otherwise update after event
        if after_event is not None:
            host_id, cell_id, old_state, new_state = after_event

            if cell_id in self.sim_to_control_cell_map:
                if old_state == "I":
                    self.infected_hosts.remove(host_id)
                if new_state == "I":
                    self.infected_hosts.add(host_id)

            return []

        # Finally, otherwise carry out culling
        if time < self.start_time:
            return []

        print(time)

        number_infected = len(self.infected_hosts)
        number_detected = np.random.binomial(number_infected, self.detection_prob)
        hosts_detected = np.random.choice(list(self.infected_hosts), number_detected, replace=False)

        hosts_to_cull = set()
        for host_id in hosts_detected:
            hosts_to_cull.add(host_id)
            host = all_hosts[host_id]
            if host.cell_id not in self.sim_to_control_cell_map:
                print("Problem: unreachable!")
            adjacent_cells = [
                host.cell_id-self.ncols-1, host.cell_id-self.ncols, host.cell_id-self.ncols+1,
                host.cell_id-1, host.cell_id, host.cell_id+1,
                host.cell_id+self.ncols-1, host.cell_id+self.ncols, host.cell_id+self.ncols+1]
            for cell in adjacent_cells:
                if 0 <= cell < len(all_cells):
                    for test_host in all_cells[cell].hosts:
                        dist = np.sqrt(
                            (host.xpos - test_host.xpos) * (host.xpos - test_host.xpos) +
                            (host.ypos - test_host.ypos) * (host.ypos - test_host.ypos))
                        if dist < self.buffer:
                            hosts_to_cull.add(test_host.host_id)

        hosts_to_cull = list(hosts_to_cull)

        return list(zip(hosts_to_cull, ['CULL']*len(hosts_to_cull)))
