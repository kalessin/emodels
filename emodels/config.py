import os
from pathlib import Path

EMODELS_REPOSITORY = os.environ.get("EMODELS_REPOSITORY", os.path.join(str(Path.home()), ".datasets"))

EMODELS_SAVE_EXTRACT_ITEMS = bool(int(os.environ.get("EMODELS_SAVE_EXTRACT_ITEMS", False)))
EMODELS_ITEMS_DIR = os.path.join(EMODELS_REPOSITORY, "items")
EMODELS_ITEMS_FILENAME = os.environ.get("EMODELS_ITEMS_FILENAME") or ""
os.makedirs(EMODELS_ITEMS_DIR, exist_ok=True)
