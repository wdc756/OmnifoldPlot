# This is the main launch file, containing settings, documentation, and execution calls



########################################################################################################################

# Imports

########################################################################################################################



# imports.check_imports will attempt to import all packages and throw errors if any are missing
from src.util.imports import check_imports
if not check_imports():
    exit(1)

from src.util.plot import *



########################################################################################################################

# Settings

########################################################################################################################



############################################################
# Main settings
############################################################


average_sets = False
average_percents = False
average_iterations = False
average_bins = False
verbose = 1

plot_defaults(average_sets, average_percents, average_iterations, average_bins, verbose)