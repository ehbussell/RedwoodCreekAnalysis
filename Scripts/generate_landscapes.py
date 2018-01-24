# Redwood Creek Analysis
# Landscape generation functions

import os
import pyproj
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import shapefile

import raster_tools

plt.style.use("ggplot")


def print_all_options():
    """Print all possible additional options to generate_landscape function."""

    print("""
generateLandscape.py

Possible additional options:
    'quiet'                 True/False, suppress printing details, default: False
    'plots'                 True/False, whether to generate figure of map, default: True
    'map_highlight_region'  Region coordinates to highlight on map, default: None
    'map_detail'            c/l/i/h/f Level of detail in map plot, default: i
    'init_cond_numbers'     True/False, generate initial conditions using host numbers as well as 
                            host density. Default: True

""")


def create_map(host_raster, region, npark, highlight_region=None, detail="c"):
    """Create figure showing host density across landscape."""

    redw_x, redw_y = npark

    wgs84 = pyproj.Proj("+init=EPSG:4326")
    NAD83_Cali_Albers = pyproj.Proj("+init=EPSG:3310")
    
    resolution = host_raster.header_vals['cellsize']
    NODATA_value = host_raster.header_vals['NODATA_value']
    xmin = host_raster.header_vals['xllcorner']
    xmax = xmin + host_raster.header_vals['ncols']*resolution
    ymin = host_raster.header_vals['yllcorner']
    ymax = ymin + host_raster.header_vals['nrows']*resolution

    x_values = np.arange(xmin, xmax, resolution)
    y_values = np.arange(ymin, ymax, resolution)

    xx, yy = np.meshgrid(x_values, y_values)

    for i in range(len(y_values)):
        for j in range(len(x_values)):
            x, y = pyproj.transform(NAD83_Cali_Albers, wgs84, xx[i, j], yy[i, j])
            xx[i, j] = x
            yy[i, j] = y

    densities = host_raster.array
    densities[densities == NODATA_value] = np.nan
    densities_masked = np.ma.array(densities, mask=np.isnan(densities))

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create map
    base_map = Basemap(epsg=3310,
                       llcrnrlat=region[0][1], llcrnrlon=region[0][0],
                       urcrnrlat=region[1][1], urcrnrlon=region[1][0],
                       resolution=detail)

    # Basic features
    base_map.drawmapboundary(fill_color='lightblue')
    base_map.fillcontinents(color='lightgrey', lake_color='lightblue', zorder=1)
    base_map.drawcoastlines()
    base_map.drawcounties(zorder=10, linestyle='dashed')

    # Add host density
    xmin_wgs, ymin_wgs = pyproj.transform(NAD83_Cali_Albers, wgs84, xmin, ymin)
    xmax_wgs, ymax_wgs = pyproj.transform(NAD83_Cali_Albers, wgs84, xmax, ymax)

    xmin_map, ymin_map = base_map(xmin_wgs, ymin_wgs)
    xmax_map, ymax_map = base_map(xmax_wgs, ymax_wgs)

    cmap = plt.get_cmap("plasma")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cax = plt.imshow(densities_masked, extent=[xmin_map, xmax_map, ymin_map, ymax_map],
                     zorder=15, alpha=0.6, cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax)

    # Add REDW NP
    poly = Polygon([base_map(x, y) for x, y in zip(redw_x, redw_y)], facecolor='green',
                   edgecolor='darkgreen', alpha=0.2, linewidth=3, zorder=20)
    ax.add_patch(poly)

    # Add highlighted region (if required)
    if highlight_region is not None:
        region_x = [highlight_region[0][0]]*2 + [highlight_region[1][0]]*2
        region_y = [highlight_region[0][1]] + [highlight_region[1][1]]*2 + [highlight_region[0][1]]

        poly2 = Polygon([base_map(x, y) for x, y in zip(region_x, region_y)],
                        edgecolor='red', alpha=0.2, linewidth=3, zorder=25)
        ax.add_patch(poly2)

    # Add SODMAP positives
    query = "Latitude > " + str(region[0][1]) + " & Latitude < " + str(region[1][1])
    query += " & Longitude > " + str(region[0][0]) + " & Longitude < " + str(region[1][0])
    query += " & State == 'Positive' & Date < '2010-07-01'"
    sodmap_df = pd.read_csv(os.path.join("InputData", "sodmap.csv"), parse_dates=True)
    region_df = sodmap_df.query(query)

    positive_lat = region_df['Latitude'].values
    positive_lon = region_df['Longitude'].values
    base_map.plot(positive_lon, positive_lat, 'x', latlon=True, color="k",
                  markersize=5, zorder=30)

    return fig


def create_initial_conditions(host_array, out_stub="InitialConditions", seed_inf_cell=(0, 0),
                              host_numbers=False, prop_infected=1.0):

    init_s_array = np.zeros((host_array.header_vals['nrows'], host_array.header_vals['ncols']))
    init_i_array = np.zeros((host_array.header_vals['nrows'], host_array.header_vals['ncols']))
    init_r_array = np.zeros((host_array.header_vals['nrows'], host_array.header_vals['ncols']))

    for row in range(host_array.header_vals['nrows']):
        for col in range(host_array.header_vals['ncols']):
            if host_array.array[row, col] > 0:
                if (row, col) != seed_inf_cell:
                    if host_numbers:
                        init_s_array[row, col] = host_array.array[row, col]
                    else:
                        init_s_array[row, col] = 1.0
                else:
                    if host_numbers:
                        init_i_array[row, col] = int(np.ceil(
                            prop_infected * host_array.array[row, col]))
                        init_s_array[row, col] = int(np.floor(
                            (1 - prop_infected) * host_array.array[row, col]))
                    else:
                        init_i_array[row, col] = prop_infected
                        init_s_array[row, col] = 1 - prop_infected

    init_s_raster = raster_tools.RasterData(
        array=init_s_array, cellsize=host_array.header_vals['cellsize'], shape=init_s_array.shape,
        llcorner=(host_array.header_vals['xllcorner'], host_array.header_vals['yllcorner']))
    init_i_raster = raster_tools.RasterData(
        array=init_i_array, cellsize=host_array.header_vals['cellsize'], shape=init_i_array.shape,
        llcorner=(host_array.header_vals['xllcorner'], host_array.header_vals['yllcorner']))
    init_r_raster = raster_tools.RasterData(
        array=init_r_array, cellsize=host_array.header_vals['cellsize'], shape=init_r_array.shape,
        llcorner=(host_array.header_vals['xllcorner'], host_array.header_vals['yllcorner']))

    init_s_raster.to_file(os.path.join("GeneratedData", out_stub + "_S.txt"))
    init_i_raster.to_file(os.path.join("GeneratedData", out_stub + "_I.txt"))
    init_r_raster.to_file(os.path.join("GeneratedData", out_stub + "_R.txt"))


def generate_landscape(region, resolution, name, options=None, print_options=False):
    """Create necessary host files for a given landscape region.

    Arguments:
        region:         Tuple of coordinates for llcorner and urcorner, each (long, lat)
        resolution:     Required resolution of output raster
        name:           Output landscape name for file outputs
        options:        Dictionary of additional options.
        print_options:  If True print all possible additional options to screen and exit
    """

    if print_options:
        print_all_options()
        return None

    if options is None:
        options = {}
    
    # Set default options
    if 'quiet' not in options:
        options['quiet'] = False
    if 'plots' not in options:
        options['plots'] = True
    if 'map_highlight_region' not in options:
        options['map_highlight_region'] = None
    if 'map_detail' not in options:
        options['map_detail'] = 'i'
    if 'init_cond_numbers' not in options:
        options['init_cond_numbers'] = True

    # Change coordinates to same format as raster
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    NAD83_Cali_Albers = pyproj.Proj("+init=EPSG:3310")
    nps_proj = pyproj.Proj("+init=EPSG:4269")

    llcorner_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *region[0])
    urcorner_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *region[1])

    # Extract host density
    host_raster = raster_tools.extract_raster(
        os.path.join("InputData", "combinedHostsScenario0.txt"), llcorner_3310, urcorner_3310,
        resolution=resolution)

    # Make landscape directory
    os.makedirs(os.path.join("GeneratedData", name), exist_ok=True)

    # Save host density raster to file
    host_raster.to_file(os.path.join("GeneratedData", name, "HostDensity.txt"))

    # Print size of raster
    if not options['quiet']:
        print("Size of " + name + " raster: {0}x{1}".format(
            host_raster.header_vals['nrows'], host_raster.header_vals['ncols']))

    # Extract host numbers
    host_num_raster = raster_tools.extract_raster(
        os.path.join("InputData", "combinedHostsScenario0.txt"), llcorner_3310, urcorner_3310,
        resolution=resolution)

    host_num_raster.array = np.multiply(host_num_raster.array,
                                        np.where(host_num_raster.array >= 0, 100, 1)).astype(int)

    host_num_raster.to_file(os.path.join("GeneratedData", name, "HostNumbers.txt"))

    # Get REDW NP shape
    sf = shapefile.Reader(os.path.join("InputData", "nps_boundary", "nps_boundary"))
    park_names = [x.record[2] for x in sf.shapeRecords()]
    redw_idx = [i for i, x in enumerate(park_names) if "Redwood" in x][0]
    redw_shp = sf.shapes()[redw_idx]

    redw_x_4269 = [x[0] for x in redw_shp.points]
    redw_y_4269 = [x[1] for x in redw_shp.points]

    # Convert REDW to lat/long
    redw_x, redw_y = pyproj.transform(nps_proj, wgs84, redw_x_4269, redw_y_4269)

    # Create region files
    # Not yet implemented

    # Create plot of host landscape
    if options['plots']:
        fig = create_map(host_raster, region, (redw_x, redw_y), options['map_highlight_region'],
                         detail=options['map_detail'])
        fig.savefig(os.path.join("Figures", name + "_map.png"), dpi=1200)

    # Generate initial conditions
    # Find location of first SODMAP positive in region
    query = "Latitude > " + str(region[0][1]) + " & Latitude < " + str(region[1][1])
    query += " & Longitude > " + str(region[0][0]) + " & Longitude < " + str(region[1][0])
    query += " & State == 'Positive'"
    sodmap_df = pd.read_csv(os.path.join("InputData", "sodmap.csv"), parse_dates=True)
    region_df = sodmap_df.query(query).sort_values("Date")

    positive_pos = (region_df['Longitude'].values[0], region_df['Latitude'].values[0])

    # Convert to map projection
    positive_pos_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *positive_pos)

    # Find cell in each raster
    cell_pos = raster_tools.find_position_in_raster(positive_pos_3310, host_raster)

    # Create initial condition files
    if options['init_cond_numbers']:
        create_initial_conditions(
            host_num_raster, out_stub=os.path.join(name, "InitialConditions_Numbers"),
            seed_inf_cell=cell_pos,
            prop_infected=(250/resolution)*(250/resolution), host_numbers=True)
    create_initial_conditions(
        host_raster, out_stub=os.path.join(name, "InitialConditions_Density"),
        seed_inf_cell=cell_pos,
        prop_infected=(250/resolution)*(250/resolution), host_numbers=False)
