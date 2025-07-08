# This file checks if the user has all required libraries

def check_imports():
    try:
        import uproot
    except ModuleNotFoundError:
        raise Exception("Import Error: missing uproot library. Check your venv or call pip install uproot")

    try:
        import numpy
    except ModuleNotFoundError:
        raise Exception("Import Error: missing numpy library. Check your venv or call pip install numpy")

    try:
        import matplotlib
    except ModuleNotFoundError:
        raise Exception("Import Error: missing matplotlib library. Check your venv or call pip install matplotlib")

    try:
        import startrace
    except ModuleNotFoundError:
        raise Exception("Import Error: missing StarTrace library. Check your venv or call pip install startrace")

    return True