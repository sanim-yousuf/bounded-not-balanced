import os
import sys
import tempfile
from pathlib import Path

# Keep every test artefact in a throw-away folder, also on Kaggle.
_TMP = Path(tempfile.mkdtemp(prefix="bbd_tests_"))
os.environ["BBD_WORK"] = str(_TMP / "work")
os.environ["BBD_INPUT"] = str(_TMP / "input")
os.environ.setdefault("BBD_TRAIN_SIZE", "1500")
os.environ.setdefault("BBD_MAX_EPOCHS", "6")
os.environ.setdefault("BBD_PATIENCE", "3")
os.environ.setdefault("BBD_N_TEST", "60")
os.environ.setdefault("BBD_N_VAL", "30")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
