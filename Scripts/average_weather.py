"""Functions for averaging forest mask and weather data."""

import logging
import os
import numpy as np
import raster_tools

def average_mask(target_header):
    """Average weather and forest type mask over full 18 years of data for given target."""

    analysis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    weather_stub = os.path.join(analysis_path, "InputData", "weather", "gis_m_c_")
    llcorner = (target_header['xllcorner'], target_header['yllcorner'])
    urcorner = (
        target_header['xllcorner'] + (target_header['ncols']) * target_header['cellsize'] - 125,
        target_header['yllcorner'] + (target_header['nrows']) * target_header['cellsize'] - 125)

    forest_mask = raster_tools.extract_raster(
        os.path.join(analysis_path, "InputData", "forestType_Map.txt"), llcorner, urcorner)
    forest_mask.array[forest_mask.array == -9999] = np.nan

    average_array = raster_tools.RasterData(
        shape=(target_header['nrows'], target_header['ncols']),
        llcorner=(target_header['xllcorner'], target_header['yllcorner']),
        cellsize=target_header['cellsize'],
        NODATA_value=target_header['NODATA_value'],
        array=np.zeros((target_header['nrows'], target_header['ncols']))
    )

    for year in range(1990, 2008):
        for week in range(1, 29):
            # Extract correct region of weather file
            weather_raster = raster_tools.extract_raster(
                weather_stub + str(year) + "_" + str(week) + ".txt", llcorner, urcorner)

            weather_raster.array[weather_raster.array == -9999] = np.nan
            if week < 7:
                weather_raster.array[forest_mask.array == 2] = 0

            if target_header['cellsize'] == weather_raster.header_vals['cellsize']:
                average_array.array += np.square(weather_raster.array)
            else:
                weather_raster.array = np.square(weather_raster.array)
                averaged_weather_array = raster_tools.extract_raster(
                    weather_raster, llcorner, urcorner, resolution=target_header['cellsize'])
                averaged_weather_array.array[averaged_weather_array.array == -9999] = 0
                average_array.array += averaged_weather_array.array

    average_array.array /= (28*18)
    average_array.array = np.sqrt(average_array.array)
    logging.info("Overall mean mask: %f", np.nanmean(average_array.array))

    average_array.array[
        np.where(np.isnan(average_array.array))] = average_array.header_vals['NODATA_value']

    return average_array
