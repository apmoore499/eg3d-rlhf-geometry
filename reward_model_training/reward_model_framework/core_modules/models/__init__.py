from .modules_aw98 import *
from .base import *
from .modules_coatnet import *
from .modules_conv3d import *
from .modules_curvenet import *
from .utils_curvenet import *
from .modules_depthmap import *
from .utils_base import *
from .modules_pointnet import *
from .modules_sigma import *
from .utils_transformer import *
from .modules_unet3d import *
from .modules_rgb import *


__all__ = [
    "modules_aw98",
    "base",
    "modules_coatnet",
    "modules_conv3d",
    "modules_curvenet",
    "utils_curvenet",
    "modules_depthmap",
    "utils_base",
    "modules_pointnet",
    "modules_sigma",
    "utils_transformer",
    "modules_unet3d",
    "modules_rgb",
]


# import glob
# import os


# #from . import external

# import sys


# import sys
# #sys.path.append(os.path.join(os.path.dirname(__file__),'external'))


# import os
# import sys

# # Assuming you want to add an 'external' directory located in the same directory as your current script
# current_script_dir = os.path.dirname(os.path.realpath(__file__))
# external_dir = os.path.join(current_script_dir, 'external')

# sys.path.append(external_dir)

# #print('pausing here')

# #import external

# from . import external

# #external import vgg

# # get all python files in the current directory
# modules = glob.glob(os.path.dirname(__file__) + "/*.py")

# # exclude __init__.py from the list of modules
# __all__ = [os.path.basename(f)[:-3] for f in modules if not f.endswith("__init__.py")]
