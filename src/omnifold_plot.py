# This is the main launch file, containing settings, documentation, and execution calls

"""

Plotting steps:

1. Get nat and syn data, for bin mapping
2. Get nat and syn weights
3. Get omnifold weights
4. Multiply syn weights by omnifold weights
5. Bin unweighted nat and syn data, to get bin mappings
6. Sum nat weights and syn weights
7. Calculate percent error between nat and syn weights
8. Apply bins to % error calculations
9. Plot each point's data and save

"""

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
average_tests = False
average_iterations = False
average_bins = True

plot_defaults(average_sets, average_percents, average_tests, average_iterations, average_bins)