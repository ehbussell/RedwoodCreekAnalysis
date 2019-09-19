"""Structure for storing and analysing fit tests on raster models."""

import os
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from scipy.spatial.distance import pdist, squareform
import raster_tools
from RasterModel import raster_model
from Scripts import MainOptions

class TestedFit:
    """Class for storing results of fit testing for raster models."""

    def __init__(self, landscape_name, base_host_numbers, run_host_numbers, test_times, sim_stub):
        self.landscape_name = landscape_name
        self.base_host_numbers = base_host_numbers
        self.run_host_numbers = run_host_numbers
        self.test_times = test_times
        self.sim_stub = sim_stub

        self.kernel_names = []
        self.cell_data = {}
        self.metric_data = {}
        self.time_data = {}
        self.kernel_generators = {}
        self.fits = {}

        self.thin_scheme = None
        self.rogue_scheme = None

    def save(self, filename):
        with open(filename, "wb") as fout:
            pickle.dump(self, fout)

    def add_kernel(self, kernel_name, kernel_generator, cell_data, all_metrics, time_data, fit):
        """Add tested kernel result for storage/analysis."""

        if kernel_name in self.kernel_names:
            raise ValueError("Kernel name already exists!")

        self.kernel_names.append(kernel_name)
        self.kernel_generators[kernel_name] = kernel_generator
        self.fits[kernel_name] = fit
        self.cell_data[kernel_name] = cell_data
        self.metric_data[kernel_name] = all_metrics
        self.time_data[kernel_name] = time_data

    def get_model(self, kernel_name):
        """Construct and return approximate model"""

        stub = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

        dimensions = (self.run_host_numbers.header_vals['nrows'],
                      self.run_host_numbers.header_vals['ncols'])
        ncells = np.product(dimensions)

        max_hosts = int(100 * np.power(self.run_host_numbers.header_vals['cellsize'] / 250, 2))
        control_scaling = np.power(2500 / self.run_host_numbers.header_vals['cellsize'], 2)

        params = {
            'inf_rate': 1.0,
            'control_rate': MainOptions.OPTIONS['control_rate'] * control_scaling,
            'coupling': np.zeros(dimensions),
            'times': self.test_times,
            'max_hosts': max_hosts,
            'primary_rate': 0.0
        }

        host_file = os.path.join(
            stub, 'GeneratedData/' + self.landscape_name + '/HostDensity.txt')
        init_s_file = os.path.join(
            stub, 'GeneratedData/' + self.landscape_name + '/InitialConditions_Density_S.txt')
        init_i_file = os.path.join(
            stub, 'GeneratedData/' + self.landscape_name + '/InitialConditions_Density_I.txt')

        model = raster_model.RasterModel(params, host_file, init_s_file, init_i_file)

        sus_file = os.path.join(
            stub, 'GeneratedData/' + self.landscape_name + '/RMSMask.txt')
        sus_raster = raster_tools.RasterData.from_file(sus_file)
        susceptibility = np.clip(sus_raster.array, 0, None).flatten()
        inf_raster = raster_tools.RasterData.from_file(sus_file)
        infectiousness = np.clip(inf_raster.array, 0, None).flatten()

        x = np.arange(ncells) % dimensions[1]
        y = np.array(np.arange(ncells) / dimensions[1], dtype=int)
        locs = np.array(list(zip(x, y)))
        dist_condensed = pdist(locs)
        distances = squareform(dist_condensed)

        kernel = self.kernel_generators[kernel_name](**self.fits[kernel_name])
        new_coupling = kernel(distances)
        model.params['coupling'] = new_coupling * infectiousness * susceptibility[:, np.newaxis]

        return model

    def video(self, kernel_name, run_num=0, video_length=5, save_file=None):
        """Plot video of simulation and ODE model runs."""

        # Read in simulation and model data
        model_data, sim_div_data, sim_agg_data = self._read_data(kernel_name)

        agg_dimensions = (self.run_host_numbers.header_vals['nrows'],
                          self.run_host_numbers.header_vals['ncols'])
        base_dimensions = (self.base_host_numbers.header_vals['nrows'],
                           self.base_host_numbers.header_vals['ncols'])

        size = (*base_dimensions, len(self.test_times))
        size_ode = (*agg_dimensions, len(self.test_times))
        nhosts_ode = int(100 * np.power(self.run_host_numbers.header_vals['cellsize'] / 250, 2))

        host_numbers = np.clip(self.base_host_numbers.array, 0, None)

        results_i = np.reshape(sim_div_data[run_num], size)
        results_s = host_numbers - results_i

        colours = np.zeros(*base_dimensions, (len(self.test_times), 4))
        colours[:, :, :, 0] = results_i/100
        colours[:, :, :, 1] = results_s/100
        colours[:, :, :, 3] = np.ones(size)

        results_i_ode = model_data
        results_s_ode = np.clip(self.run_host_numbers.array, 0, None) - model_data

        colours_ode = np.zeros((*agg_dimensions, len(self.test_times), 4))
        colours_ode[:, :, :, 0] = results_i_ode/nhosts_ode
        colours_ode[:, :, :, 1] = results_s_ode/nhosts_ode
        colours_ode[:, :, :, 3] = np.ones(size_ode)

        video_length *= 1000
        nframes = size[0]

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig.tight_layout()

        im1 = ax1.imshow(colours[:, :, 0], animated=True, origin="upper")
        im2 = ax2.imshow(colours_ode[:, :, 0], animated=True, origin="upper")
        time_text = ax1.text(0.03, 0.9, 'time = %.3f' % self.test_times[0],
                             transform=ax1.transAxes, weight="bold", fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.6))

        def update(frame_number):
            im1.set_array(colours[:, :, frame_number])
            im2.set_array(colours_ode[:, :, frame_number])
            time_text.set_text('time = %.3f' % self.test_times[frame_number])

            return im1, im2, time_text

        im_ani = animation.FuncAnimation(fig, update, interval=video_length/nframes, frames=nframes,
                                         blit=True, repeat=False)

        if save_file is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            im_ani.save(save_file+'.mp4', writer=writer)

        plt.show()

    def interact(self, kernel_name):
        """Generate interactive plot for analysing the tested fit for that kernel."""

        # Read in simulation and model data
        model_data, sim_div_data, sim_agg_data = self._read_data(kernel_name)

        # Make mapping to aggregate full resolution simulaton cells
        agg_map = raster_tools.aggregate_cells(
            self.base_host_numbers.header_vals, self.run_host_numbers.header_vals,
            generate_reverse=False, ignore_outside_target=True)

        if kernel_name not in self.kernel_names:
            raise ValueError("Kernel does not exist!")

        # Custom colormap
        cdict1 = {
            'red':      ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),

            'green':    ((0.0, 0.0, 0.0),
                         (1.0, 1.0, 1.0)),

            'blue':     ((0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0))
        }

        bl_gr = mpl.colors.LinearSegmentedColormap("bl_gr", cdict1)

        fig = plt.figure()

        # Divided landscape plot
        xtick_max = self.base_host_numbers.header_vals["ncols"] - 0.5
        ytick_max = self.base_host_numbers.header_vals["nrows"] - 0.5
        tick_step = (self.run_host_numbers.header_vals["cellsize"] /
                     self.base_host_numbers.header_vals["cellsize"])
        ax1 = fig.add_subplot(221)
        cax1 = ax1.imshow(
            self.base_host_numbers.array, interpolation="nearest", cmap=bl_gr, vmin=0, vmax=100)
        ax1.set_title("Divided Landscape\nLandscape RMSE value: {:.4G}".format(
            self.metric_data[kernel_name]["Landscape"]))
        ax1.set_xticks(np.arange(-0.5, xtick_max, tick_step))
        ax1.set_xticklabels([])
        ax1.set_yticks(np.arange(-0.5, ytick_max, tick_step))
        ax1.set_yticklabels([])

        # Divided RMSE map
        ax3 = fig.add_subplot(222)
        cax3 = ax3.imshow(self.cell_data[kernel_name]["Divided"], interpolation="nearest",
                          cmap="hot")
        cbar3 = fig.colorbar(cax3, ax=ax3)
        cbar3.set_label("Cell RMSE value")
        ax3.set_title("Divided RMSE map\nDivided RMSE value: {:.4G}".format(
            self.metric_data[kernel_name]["Divided"]))
        ax3.set_xticks(np.arange(-0.5, xtick_max, tick_step))
        ax3.set_xticklabels([])
        ax3.set_yticks(np.arange(-0.5, ytick_max, tick_step))
        ax3.set_yticklabels([])

        # Normal landscape plot
        xtick_max = self.run_host_numbers.header_vals["ncols"] - 0.5
        ytick_max = self.run_host_numbers.header_vals["nrows"] - 0.5
        ax2 = fig.add_subplot(223)
        cax2 = ax2.imshow(
            self.run_host_numbers.array, interpolation="nearest", cmap=bl_gr, vmin=0,
            vmax=100)
        ax2.set_title("Run Landscape")
        ax2.set_xticks(np.arange(-0.5, xtick_max, 1.0))
        ax2.set_xticklabels([])
        ax2.set_yticks(np.arange(-0.5, ytick_max, 1.0))
        ax2.set_yticklabels([])

        # Normal RMSE map
        ax4 = fig.add_subplot(224)
        cax4 = ax4.imshow(self.cell_data[kernel_name]["Normal"], interpolation="nearest",
                          cmap="hot")
        cbar4 = fig.colorbar(cax4, ax=ax4)
        cbar4.set_label("Cell RMSE value")
        ax4.set_title("Normal RMSE map\nNormal RMSE value: {:.4G}".format(
            self.metric_data[kernel_name]["Normal"]))
        ax4.set_xticks(np.arange(-0.5, xtick_max, 1.0))
        ax4.set_xticklabels([])
        ax4.set_yticks(np.arange(-0.5, ytick_max, 1.0))
        ax4.set_yticklabels([])

        fig.suptitle("{0}: {1}".format(self.landscape_name, kernel_name))

        def _onclick(event):
            if event.inaxes == ax1 or event.inaxes == ax2:
                aggregation = "Landscape"
            elif event.inaxes == ax3:
                aggregation = "Divided"
            elif event.inaxes == ax4:
                aggregation = "Normal"
            else:
                return

            if aggregation != "Landscape":
                row = np.around(event.ydata)
                col = np.around(event.xdata)
                cell_id = int(col) + (int(row) * self.cell_data[kernel_name][aggregation].shape[1])
            else:
                cell_id = None
                self._plot_dpc(aggregation, cell_id, model_data, sim_agg_data, agg_map)

            if aggregation == "Divided":
                self._plot_dpc(aggregation, cell_id, model_data, sim_div_data, agg_map)
            if aggregation == "Normal":
                self._plot_dpc(aggregation, cell_id, model_data, sim_agg_data, agg_map)

        cid = fig.canvas.mpl_connect('button_press_event', _onclick)
        plt.show()

    def _plot_dpc(self, aggregation, cell_id, model_data, sim_data, agg_map):

        figi = plt.figure()
        ax = figi.add_subplot(111)

        if aggregation == "Landscape":
            for sim in sim_data:
                ax.plot(self.test_times, np.sum(sim, axis=0), "r", alpha=0.2)
            ax.plot(self.test_times, np.sum(model_data, axis=0), "b", lw=2)

        if aggregation == "Normal":
            for sim in sim_data:
                ax.plot(self.test_times, sim[cell_id, :], "r", alpha=0.2)
            ax.plot(self.test_times, model_data[cell_id, :], "b", lw=2)

        if aggregation == "Divided":
            for sim in sim_data:
                ax.plot(self.test_times, sim[cell_id, :], "r", alpha=0.2)

            # Find correct cell of model run
            agg_cell_id = agg_map[cell_id]
            ncells = np.power(
                self.run_host_numbers.header_vals['cellsize'] /
                self.base_host_numbers.header_vals['cellsize'], 2)
            ax.plot(self.test_times, model_data[agg_cell_id, :] / ncells, "b", lw=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Number Infected")

        if cell_id is not None:
            ax.set_title("Cell {0}".format(cell_id))
        else:
            ax.set_title("Landscape")
        figi.show()

    def _read_data(self, kernel_name):
        """Read in simulation data, and run approximate model."""

        # First generate approximate model data
        model = self.get_model(kernel_name)
        dimensions = (self.run_host_numbers.header_vals['nrows'],
                      self.run_host_numbers.header_vals['ncols'])
        ncells = np.product(dimensions)

        no_control_tmp = model.run_scheme(self.thin_scheme, self.rogue_scheme)
        model_data = np.zeros((ncells, len(self.test_times)))
        for cell in range(ncells):
            model_data[cell] = no_control_tmp.results_i['Cell' + str(cell)].values

        # Now extract simulation data
        # At full resolution
        filename = os.path.join(self.sim_stub + "summaries", "ROI_250Landscape" + ".h5")
        with h5py.File(filename, 'r') as hf:
            sim_div_data = hf['sim_summary_I'][:]

        # And at approximate model resolution
        filename = os.path.join(self.sim_stub + "summaries", self.landscape_name + ".h5")
        with h5py.File(filename, 'r') as hf:
            sim_agg_data = hf['sim_summary_I'][:]

        return model_data, sim_div_data, sim_agg_data
