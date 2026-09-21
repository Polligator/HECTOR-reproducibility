"""Single source of truth for project paths used by the plot package."""
from pathlib import Path

# scripts/model_eval/plot/paths.py -> parents[3] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "result"        # figures land here (figure_4.*, supp_s*.pdf)
DATA_DIR = RESULT_DIR / "data"              # csv/pkl inputs the figure reads
PLOT_CACHE_DIR = DATA_DIR / "plot_cache"    # relocated panel caches
ONTOLOGY_OBO = PROJECT_ROOT / "input" / "cl.obo"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
