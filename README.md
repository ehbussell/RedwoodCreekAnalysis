# Redwood Creek Analysis

Code for analysing invasion of sudden oak death into Redwood National Park in California.

## Requirements
This project relies on code from [SpatialSimulator](https://github.com/ehbussell/SpatialSimulator) repo, and [RasterModel](https://github.com/ehbussell/RasterModel) repo. Optimisation in the raster model also requires [Ipopt](https://github.com/coin-or/Ipopt).

Other data is required in InputData directory - most importantly host density file, weather data and forest type map from Meentemeyer *et al/.* (2011). These are currently unavailable.

## Overall analysis approach

1. Create simulation data
    * Using kernel from Meentemeyer *et al.* (2011) and virtual sporulation generate a raster kernel at 250m resolution
    * From host density file construct individual landscape with hosts at cell centres
    * Run simulation on this landscape.
1. Fit raster-based ODE model
    * Using simulation data, fit possible kernels in raster model
    * Select between kernels based on RMSE metric
1. Test how quality of fit varies with resolution
1. Find optimal control
1. Test optimal control against strategy carried out in practice (100m buffer)

## File description
Whole analysis can be run using ``run_all.py`` but note: will be slow. Files in Scripts folder investigate individual effects.