import os
from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parent
TMP_DIR = REPO_ROOT / "temp" / "colab"


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_ROOT)
    os.environ.setdefault("SIMBAY_HEADLESS", "1")
    os.environ.setdefault("SIMBAY_USE_MJX", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("SIMBAY_PARTICLES", "100")
    os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(TMP_DIR / "xdg-cache"))
    runpy.run_path(str(REPO_ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
