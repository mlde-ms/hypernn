import sys, os
from pathlib import Path

def configure_path():
    current_path = Path(os.getcwd())
    parent_path = current_path.parent

    sys.path.append('..')