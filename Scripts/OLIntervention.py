"""Intervention class script implementing open-loop control from RasterModel optimisation."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import raster_tools

class Intervention:
    """Open-loop intervention"""

    def __init__(self, update_freq, all_hosts, all_cells, options):

        control_raster_file, sim_raster_file, control_rate, results_stub = options
        control_raster_header = raster_tools.RasterData.from_file(control_raster_file).header_vals
        sim_raster_header = raster_tools.RasterData.from_file(sim_raster_file).header_vals

        self.n_control_cells = control_raster_header['nrows'] * control_raster_header['ncols']

        # Construct map from simulation cell id to control cell id
        self.sim_to_control_cell_map, self.control_to_sim_cell_map = raster_tools.aggregate_cells(
            sim_raster_header, control_raster_header, generate_reverse=True,
            ignore_outside_target=True)

        # Set required properties: update frequency, intervention type, & rate structure size
        self.update_freq = update_freq
        self.type = "CONTINUOUS"
        self.rate_size = 2 * self.n_control_cells
        self.rate_factor = float(control_rate)

        self.infected_hosts = np.zeros(self.n_control_cells, dtype=int)
        self.susceptible_hosts = np.zeros(self.n_control_cells, dtype=int)
        self.rogue_level = np.zeros(self.n_control_cells)
        self.thin_level = np.zeros(self.n_control_cells)

        results_rogue = pd.read_csv(results_stub + "_v.csv")
        results_thin = pd.read_csv(results_stub + "_u.csv")

        self.rogue_scheme = interp1d(results_rogue['time']*28, results_rogue.values[:, 1:].T,
                                     kind="zero", fill_value="extrapolate")
        self.thin_scheme = interp1d(results_thin['time']*28, results_thin.values[:, 1:].T,
                                    kind="zero", fill_value="extrapolate")


    def update(self, all_hosts, time, all_cells=None, after_event=None, get_rate_fn=None,
               initial=False):

        if initial:
            # Perform initial setup
            for cell in all_cells:
                control_cell_id = self.sim_to_control_cell_map.get(cell.cell_id, None)
                if control_cell_id is not None:
                    self.infected_hosts[control_cell_id] += cell.states['C'] + cell.states['I']
                    self.susceptible_hosts[control_cell_id] += cell.states['S']

            self.rogue_level = self.rogue_scheme(time)
            self.thin_level = self.thin_scheme(time)

            rate_updates = [(2*cell, self.rogue_level[cell] * self.infected_hosts[cell])
                            for cell in range(self.n_control_cells)]
            rate_updates += [(2*cell+1, self.thin_level[cell] * self.susceptible_hosts[cell])
                             for cell in range(self.n_control_cells)]

            return rate_updates

        # Otherwise update after event
        if after_event is not None:
            host_id, cell_id, old_state, new_state = after_event
            control_cell_id = self.sim_to_control_cell_map.get(cell_id, None)
            if control_cell_id is None:
                return []

            if old_state == "S":
                self.susceptible_hosts[control_cell_id] -= 1

                if new_state in "CI":
                    self.infected_hosts[control_cell_id] += 1

            elif (old_state in "CI") and (new_state not in "CI"):
                self.infected_hosts[control_cell_id] -= 1

            rate_updates = [
                (2*control_cell_id,
                 self.rogue_level[control_cell_id] * self.infected_hosts[control_cell_id]),
                (2*control_cell_id+1,
                 self.thin_level[control_cell_id] * self.susceptible_hosts[control_cell_id])
            ]

            return rate_updates

        # Finally, otherwise update rogue and thin levels
        self.rogue_level = self.rogue_scheme(time)
        self.thin_level = self.thin_scheme(time)

        rate_updates = [(2*cell, self.rogue_level[cell] * self.infected_hosts[cell])
                        for cell in range(self.n_control_cells)]
        rate_updates += [(2*cell+1, self.thin_level[cell] * self.susceptible_hosts[cell])
                         for cell in range(self.n_control_cells)]

        return rate_updates

    def action(self, all_hosts, time, event_id, all_cells=None):
        """Carry out cull action: either rogue or thin randomly from control cell."""

        sim_cell_ids = self.control_to_sim_cell_map[event_id // 2]
        np.random.shuffle(sim_cell_ids)

        host_id = None
        if (event_id % 2) == 0:
            # Roguing event
            for cell_id in sim_cell_ids:
                cell = all_cells[cell_id]
                if cell.states['I'] > 0:
                    for host in cell.hosts:
                        if host.state == "I":
                            host_id = host.host_id
                            break

        else:
            # Thinning event
            for cell_id in sim_cell_ids:
                cell = all_cells[cell_id]
                if cell.states['S'] > 0:
                    for host in cell.hosts:
                        if host.state == "S":
                            host_id = host.host_id
                            break

        if host_id is None:
            raise ValueError("Cannot find host to remove!\n")

        return [(host_id, 'CULL')]
