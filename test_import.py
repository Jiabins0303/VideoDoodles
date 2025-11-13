#!/usr/bin/env python3
"""Test script to verify Video-Depth-Anything import works with serialization patch"""
import sys
import types
from pathlib import Path

# Add Video-Depth-Anything to path
vda_path = Path("Video-Depth-Anything")
sys.path.insert(0, str(vda_path))

# Import torch first
import torch

# Patch torch.utils.serialization BEFORE any imports
if not hasattr(torch.utils, 'serialization'):
    # Create a dummy serialization module
    serialization_module = types.ModuleType('serialization')
    # Add dummy attributes that xformers might try to access
    serialization_module.read_lua_file = lambda *args, **kwargs: None
    serialization_module.load_lua = lambda *args, **kwargs: None
    
    # Create a dummy config object that xformers might try to import
    # xformers does: from serialization import config
    config_obj = types.ModuleType('config')
    # Add any attributes config might need (return None for any attribute access)
    config_obj.__getattr__ = lambda name: None
    serialization_module.config = config_obj
    
    # Attach to torch.utils
    torch.utils.serialization = serialization_module
    # Add to sys.modules so 'from torch.utils import serialization' works
    # Also handle 'import torch.utils.serialization' and 'from serialization import config'
    sys.modules['torch.utils.serialization'] = serialization_module
    sys.modules['serialization'] = serialization_module  # For direct 'from serialization import config'
    # Make sure torch.utils is in sys.modules too
    if 'torch.utils' not in sys.modules:
        sys.modules['torch.utils'] = torch.utils

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Now try to import Video-Depth-Anything
try:
    from video_depth_anything.video_depth import VideoDepthAnything
    print("✓ SUCCESS: Video-Depth-Anything imported successfully!")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Device available: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    sys.exit(0)
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

