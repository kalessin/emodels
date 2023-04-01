import os
from pathlib import Path

EMOÐELS_DIR = os.environ.get("EMOÐELS_DIR", os.path.join(str(Path.home()), ".datasets"))
