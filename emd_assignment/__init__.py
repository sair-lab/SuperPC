import sys
import os

# Add the emd_assignment directory to the Python path so emd module can be found
_emd_dir = os.path.dirname(os.path.abspath(__file__))
if _emd_dir not in sys.path:
    sys.path.insert(0, _emd_dir)

# Import the emd_module
from . import emd_module
