"""Structure for storing and analysing fit tests on raster models."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class TestedFit:
    """Class for storing results of fit testing for raster models."""

    def __init__(self, landscape_name, base_raster, run_raster, all_sim_data, test_times):
        self.landscape_name = landscape_name
        self.base_raster = base_raster
        self.run_raster = run_raster
        self.sim_data = all_sim_data
        self.test_times = test_times

        self.kernel_names = []
        self.run_data = {}
        self.cell_data = {}
        self.metric_data = {}

    def save(self, filename):
        with open(filename, "wb") as fout:
            pickle.dump(self, fout)

    def add_kernel(self, kernel_name, all_run_data, cell_data, all_metrics):
        """Add tested kernel result for storage/analysis."""

        if kernel_name in self.kernel_names:
            raise ValueError("Kernel name already exists!")

        self.kernel_names.append(kernel_name)
        self.run_data[kernel_name] = all_run_data
        self.cell_data[kernel_name] = cell_data
        self.metric_data[kernel_name] = all_metrics

    def interact(self, kernel_name):
        """Generate interactive plot for analysing the tested fit for that kernel."""

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
        xtick_max = self.base_raster.header_vals["ncols"] - 0.5
        ytick_max = self.base_raster.header_vals["nrows"] - 0.5
        tick_step = (self.run_raster.header_vals["cellsize"] /
                     self.base_raster.header_vals["cellsize"])
        ax1 = fig.add_subplot(221)
        cax1 = ax1.imshow(
            self.base_raster.array, interpolation="nearest", cmap=bl_gr, vmin=0, vmax=1)
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
        xtick_max = self.run_raster.header_vals["ncols"] - 0.5
        ytick_max = self.run_raster.header_vals["nrows"] - 0.5
        ax2 = fig.add_subplot(223)
        cax2 = ax2.imshow(
            self.run_raster.array, interpolation="nearest", cmap=bl_gr, vmin=0, vmax=1)
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

            self._plot_dpc(kernel_name, aggregation, cell_id)

        cid = fig.canvas.mpl_connect('button_press_event', _onclick)
        plt.show()

    def _plot_dpc(self, kernel_name, aggregation, cell_id):
        figi = plt.figure()
        ax = figi.add_subplot(111)
        if cell_id is not None:
            for sim in self.sim_data[aggregation]:
                ax.plot(self.test_times, sim[:, cell_id], "r", alpha=0.2)
            run = self.run_data[kernel_name][aggregation][cell_id]
        else:
            for sim in self.sim_data[aggregation]:
                ax.plot(self.test_times, sim, "r", alpha=0.2)
            run = self.run_data[kernel_name][aggregation]
        ax.plot(run[:, 0], run[:, 2], "b", lw=2)
        if cell_id is not None:
            ax.set_title("Cell {0}".format(cell_id))
        else:
            ax.set_title("Landscape")
        figi.show()
