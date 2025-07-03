# This file checks if the user has all required libraries

try:
    import ROOT
except ModuleNotFoundError:
    raise Exception("Import Error: missing ROOT library. Did you forget to activate root (/<root install>/bin/thisroot.bat or thisroot.sh")

try:
    import numpy
except ModuleNotFoundError:
    raise Exception("Import Error: missing numpy library. Check you venv or call pip install numpy")

try:
    import matplotlib
except ModuleNotFoundError:
    raise Exception("Import Error: missing matplotlib library. Check you venv or call pip install matplotlib")

