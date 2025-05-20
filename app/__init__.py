"""
Init script.
"""
import os
from osgeo import __file__ as f

# Define Proj database location
os.environ["PROJ_LIB"] = os.path.join(os.path.dirname(f), 'data', 'proj')
