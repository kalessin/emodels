import os
from pathlib import Path

EMODELS_DIR = os.environ.get("EMODELS_DIR", os.path.join(str(Path.home()), ".datasets"))
EMODELS_ENABLED = bool(int(os.environ.get("EMODELS_ENABLED", False)))

EMODELS_ITEMS_DIR = os.path.join(EMODELS_DIR, "items")
os.makedirs(EMODELS_ITEMS_DIR, exist_ok=True)
