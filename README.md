# NestModelSimplification

## Python
Tested on python 2.7 -- Should run on python 3.x with minimal (if any) fixes.

## Required libraries
*NEST: http://www.nest-simulator.org
*h5py: http://www.h5py.org
*Numpy: http://www.numpy.org

## Optional libraries
*Seaborn (Pretty Plots): `pip install seaborn`

## Steps
*Step 1: Download required neuron models Example ((https://bbp.epfl.ch/nmc-portal/microcircuit#/metype/L5_TTPC1_cADpyr/details))

*Step 2: Place scripts from here into the working NEURON model directory

*Step 3: `Run Simulator.py` -- Ensuring Optimize Flag is set to "True" %% This will be very time consuming

*Step 4: `Run modelfit.py` -- This will fit the data using GIFFittingToolBox 

*Step 5: `Run NestGIFModel.py` -- This will take the dictionary of parameters generated by Step 4 and create a basic NEST network

## Notes
Note: You will need to modify the source somewhat to set paths for data. I have made those locations as obvious as possible ex. `'### FILL IN ###'` should be changed to your path. 

Note: On the subject of goodness of fit. We are using DETERMINISTIC models, GIFFittingToolsBox uses MD (see report refrences) to calculate goodness of fit, this is NOT a appropriate metric for goodness for DETERMINISTIC models. Use Variance Exampled (see report) to assess the models fitting.  

Visualization: If you wish to add visualization to ensure your data works please edit the scripts. In modelfit this can be done by removing comments from Line 76:`# myExp.plotTrainingSet()` && Line 77:` # myExp.plotTestSet()` this will allow you to visualize your generated data. Goodness of fit will automatically be visualized (and pushed to your terminal). If you wish to visualize the current (A good idea) this can be done very inexpensivly by either modifying `CurrentGenerator.py` with the following lines or writing a small pythong script with the same lines and importing CurrentGenerator:
```python
current = CurrentGenerator(time=100000, optimize_flag=False)                         
current_vector = [x for x in current.generate_current()]                                         
plotcurrent(current_vector)
```
This will generate a simple plot of the current over 100,000ms (note: it will generate time/dt points).

Adendum: The majority of the code base is the `CurrentGenerator.py`. It should function stand-alone without modification. The Simulator will require several small edits to ensure paths && Saved data directories are correct. Please be careful that your saved directories are known to the different scripts.

## Data Flow

Dataflow:
*-> Simulator will inject a current vector into the NEURON model and save the required current/voltage response to an HDF5 formated file.
*--> Modelfit will load that dataset, fit it, and save a pickled dictionary with parameters locally
*---> NestGIFModel will use that pickled dictionary to generate the NEST GIF model of choice 

## Examples

Examples: Examples are provided in the `Examples` branch and you can see a slightly outdated by relevant variant I personally utilized here (https://github.com/cigani/NEST/blob/master/L5_TTPC1_cADpyr232_1/Simulator.py)

## Depreciated

`h5Xchange.py` is unused. Feel free to ignore it. It may be useful to you if your data is saved by numpy as plain-text (it will load and convert plain-text data to H5 data sets in the form needed to run the fitting toolbox).

# Warnings

The scripts will NOT function without editing the paths.
