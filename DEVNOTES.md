# Architecture/Upgrades

The `niftynet/handle.py` script handles the interface between the outside world and the network itself. It does:

1. Parses input command-line arguments
2. Resamples/converts the image into the proper format/spacing
3. Writes the configuration file for the network
4. Runs the network
5. Converts/post-processes (island removal) the output segmentation

When upgrading the network you'll likely have to rewrite significantly most of the steps, based on the specific requirements of the network. The configuration file bit (step 3) is a peculiarity of NiftyNet, and hopefully a future network will be run directly from Python rather than running it using a subprocess (step 4). Most of the resampling/conversion code (steps 2 and 5) will stay the same, but some file extensions will need to be changed (and maybe something other than SimpleITK will be required). When possible, the output files should be compressed.

# CUDA BSpline Resampling

Upgrading this is much trickier. The `fix_spacing` function in the `niftynet/support.py` file shouldn't need to be changed. The actual CUDA/C code is in the `cudaresize` folder; you'll notice that quite a bit is missing. The required code is downloaded from the base CUDA BSpline GitHub repository when the Docker image is built; see the Dockerfile for how this is done. 

The only change that would need to be made here is to add support for other data types; due to C++ templates being C++ templated, you have to explicitly instantiate the template for each specific data type you want to resample from Python.
