""" Redwood Creek Analysis: Landscape generation functions. """

import logging
import os
import pyproj
import numpy as np
import pandas as pd

import cartopy
from cartopy.io.shapereader import Reader
from functools import partial
from shapely.ops import transform
from shapely import geometry
import shapefile

import raster_tools
from Scripts import average_weather
from Scripts import MainOptions


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


def generate_landscape(region, resolution, name):
    """Create necessary host files for a given landscape region.

    Arguments:
        region:         Tuple of coordinates for llcorner and urcorner, each (long, lat)
        resolution:     Required resolution of output raster
        name:           Output landscape name for file outputs
    """

    analysis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    # Change coordinates to same format as raster
    wgs84 = pyproj.Proj("+init=EPSG:4326")
    NAD83_Cali_Albers = pyproj.Proj("+init=EPSG:3310")
    nps_proj = pyproj.Proj("+init=EPSG:4269")

    llcorner_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *region[0])
    urcorner_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *region[1])

    input_data_file = os.path.join(analysis_path, "InputData", "combinedHostsScenario0.txt")

    # Extract host density
    host_raster = raster_tools.extract_raster(input_data_file, llcorner_3310, urcorner_3310,
                                              resolution=resolution)

    # Make landscape directory
    os.makedirs(os.path.join("GeneratedData", name), exist_ok=True)

    # Save host density raster to file
    host_raster.to_file(os.path.join("GeneratedData", name, "HostDensity.txt"))

    logging.info("Size of %s raster: %dx%d", name, host_raster.header_vals['nrows'],
                 host_raster.header_vals['ncols'])

    # Extract host numbers
    host_num_raster = raster_tools.extract_raster(
        input_data_file, llcorner_3310, urcorner_3310, resolution=resolution)

    host_num_raster.array = np.multiply(host_num_raster.array,
                                        np.where(host_num_raster.array >= 0, 100, 1)).astype(int)

    host_num_raster.to_file(os.path.join("GeneratedData", name, "HostNumbers.txt"))

    logging.info("Starting NP mask raster")
    np_raster = make_np_mask(host_num_raster.header_vals)
    np_raster.to_file(os.path.join("GeneratedData", name, "NPMask.txt"))

    logging.info("Starting initial conditions")

    # Generate initial conditions
    # Find location of first SODMAP positive in region
    query = "Latitude > " + str(region[0][1]) + " & Latitude < " + str(region[1][1])
    query += " & Longitude > " + str(region[0][0]) + " & Longitude < " + str(region[1][0])
    query += " & State == 'Positive'"
    sodmap_df = pd.read_csv(
        os.path.join(analysis_path, "InputData", "sodmap.csv"), parse_dates=True)
    region_df = sodmap_df.query(query).sort_values("Date")

    positive_pos = (region_df['Longitude'].values[0], region_df['Latitude'].values[0])

    # Convert to map projection
    positive_pos_3310 = pyproj.transform(wgs84, NAD83_Cali_Albers, *positive_pos)

    # Find cell in each raster
    cell_pos = raster_tools.find_position_in_raster(positive_pos_3310, host_raster)

    logging.info("Found source cell")

    # Create initial condition files
    base_host_raster = raster_tools.extract_raster(
        os.path.join(analysis_path, "InputData", "combinedHostsScenario0.txt"), llcorner_3310,
        urcorner_3310, resolution=250)
    base_cell_pos = raster_tools.find_position_in_raster(positive_pos_3310, base_host_raster)
    base_inf_density = base_host_raster.array[base_cell_pos]
    ncells = (resolution/250)*(resolution/250)
    prop_inf = base_inf_density / (host_raster.array[cell_pos] * ncells)
    create_initial_conditions(
        host_num_raster, out_stub=os.path.join(name, "InitialConditions_Numbers"),
        seed_inf_cell=cell_pos, prop_infected=prop_inf, host_numbers=True)
    create_initial_conditions(
        host_raster, out_stub=os.path.join(name, "InitialConditions_Density"),
        seed_inf_cell=cell_pos, prop_infected=prop_inf, host_numbers=False)

    # Create averaged weather and forest type mask
    avg_mask = average_weather.average_mask(target_header=host_raster.header_vals)
    avg_mask.to_file(os.path.join("GeneratedData", name, "RMSMask.txt"))

def make_np_mask(target_header):
    """Generate mask of cells that are in Redwood National Park"""

    rdr = Reader(os.path.join("InputData", "nps_boundary", "nps_boundary.shp"))
    redw_records = []
    for x in rdr.records():
        if 'Redwood' in x.attributes['UNIT_NAME']:
            redw_records.append(x)

    redw_shape_nps = redw_records[0].geometry[0]

    NAD83_Cali_Albers = pyproj.Proj("+init=EPSG:3310")
    nps_proj = pyproj.Proj("+init=EPSG:4269")

    project = partial(pyproj.transform, nps_proj, NAD83_Cali_Albers)

    redw_shape = transform(project, redw_shape_nps)

    lower_x = np.array([target_header['xllcorner'] + i*target_header['cellsize']
                        for i in range(target_header['ncols'])])
    upper_x = lower_x + target_header['cellsize']

    lower_y = np.array([target_header['yllcorner'] + i*target_header['cellsize']
                        for i in range(target_header['nrows'])])[::-1]
    upper_y = lower_y + target_header['cellsize']

    np_array = np.zeros((target_header['nrows'], target_header['ncols']))

    for i in range(target_header['nrows']):
        for j in range(target_header['ncols']):
            points = [[lower_x[j], lower_y[i]], [upper_x[j], lower_y[i]], [upper_x[j], upper_y[i]],
                      [lower_x[j], upper_y[i]]]
            cell = geometry.Polygon(points)
            intersection_area = redw_shape.intersection(cell).area / (
                target_header['cellsize'] * target_header['cellsize'])

            np_array[i, j] = intersection_area

    np_raster = raster_tools.RasterData(
        (target_header['nrows'], target_header['ncols']),
        (target_header['xllcorner'], target_header['yllcorner']), target_header['cellsize'],
        array=np_array)

    return np_raster
