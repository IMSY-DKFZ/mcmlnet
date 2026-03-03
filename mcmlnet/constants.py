"""
Constants used throughout the mcmlnet package.

"""

import os

from dotenv import load_dotenv

load_dotenv()

# Unit conversion constants
NM_TO_M = 1e-9  # Convert nanometers to meters
CM_INV_TO_M_INV = 1e2  # Convert cm^-1 to m^-1
REFERENCE_WAVELENGTH_M = 500e-9  # Reference wavelength for scattering calculations
REFERENCE_WAVELENGTH_NM = 500  # Reference wavelength in nanometers
RAYLEIGH_EXPONENT = -4.0  # Rayleigh scattering exponent

# MCMLGPU environment variable configuration
MCML_PATH = os.getenv("MCML_PATH")
MCO_PATH = os.getenv("MCO_PATH")
