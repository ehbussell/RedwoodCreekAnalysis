# Redwood Creek Project Plan

## Current Questions
1. What normalisation to use for fitted raster model kernels? In particular, I think requires normalisation in 2D but is this possible for all kernels? Cauchy can't be normalised in 2D?

## Overall Plan

1. Create simulation data
    * Using kernel from Meentemeyer and virtual sporulation generate a raster kernel (positive quadrant only) at 250m resolution
    * From host density file construct individual landscape with hosts at cell centres
    * Run individual based simulation on this landscape extracting host transition times.  **Note: will require correct probability of spore successful challenge.**
1. Construct likelihood
    * Using host transition times, calculate likelihood function for required resolution ODE model
1. Fit possible kernels
    * Select between kernels based on some metric
1. Test how quality of fit varies with resolution
1. Find optimal control
1. Test optimal control and MPC against alternative strategies

## Section Details

### Create simulation data
1. Generate landscape of host unit numbers from California host landscape
1. Use kernel from Meentemeyer to extract raster kernel (positive quadrant)
1. Generate an individual based simulation with individuals snapped to centre of cells
1. Run simulations and extract individual infection times

#### To do:
* Calculate host numbers in 250m cells
* Extract raster kernel
* Possibly run sample simulations using Richard's simulator here
* Adapt individual simulator to take advantage of 'snap to grid'
* Create initial conditions
* Run sample simulations
* Run full simulation set

### Fitting Process
1. Calculate likelihood calculation
1. Use MCMC to calculate kernel parameters in new resolution model for multiple kernel types:
    * Cauchy
    * Exponential
    * Power law
    * Skelsey type (both exponential and power law)
1. Choose between kernels based on fit metric (eg wave speed, number infected etc.)

#### To do:
* Decide on kernel forms
* Decide whether to normalise and integrate kernels in raster model
* Formulate likelihood calculation for raster ODE, for each kernel type
* Decide on epidemic metric for comparing kernels (and quality of fit later)
* Choose between kernels

### Resolution Testing
1. Define extended region of interest (EROI), over which the full simulation will be run.
1. Define smaller region of interest (ROI) for fitting and optimisation.
1. Choose set of resolutions for RasterModel and optimisation.  For each resolution:
    1. Generate host density and initial condition files for ROI.
    1. Fit RasterModel.
    1. Assess quality of fit under no control.
    1. Assess quality of fit under specified control policy, with varying budget.

#### To do:
* Decide on suitable region sizes and resolutions
* Decide on control policy to implement for quality assessment

### Optimal Control Scheme

1. Choose best resolution (taking into account maximum resolution of optimiser).
1. Define regions within ROI, value region Redwood NP.
1. Set up NLP optimisation to minimise disease within value region.
1. Test optimal control scheme against other policies on full simulation model (EROI?).

#### To do:
* How to extract value region (polygon -> raster)
* Suitable NLP set-up - improve NLP program form, options etc.
* Other schemes to test against

### Model Predictive Control

1. Repeatedly run Optimal Control Scheme at regular update intervals.
1. Vary update interval.
1. Compare results with Optimal scheme, and other strategies.
1. Test how wrong other strategies & optimal scheme can be for these to be the best control.

#### To do:
* Decide on update intervals to test
* How to look at how wrong other strategies can be?