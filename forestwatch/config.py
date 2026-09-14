from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    root: Path = Path(os.getenv("FORESTWATCH_ROOT", Path(__file__).resolve().parents[1]))
    dataset_path: Path = Path(os.getenv("FORESTWATCH_DATASET", root / "dataset"))
    project_data_path: Path = Path(os.getenv("FORESTWATCH_PROJECT_DATA", root.parent / "proj"))
    output_path: Path = Path(os.getenv("FORESTWATCH_OUTPUT", root / "data"))
    low_ndvi: float = float(os.getenv("FORESTWATCH_LOW_NDVI", "0.3"))
    healthy_ndvi: float = float(os.getenv("FORESTWATCH_HEALTHY_NDVI", "0.6"))
    decrease_threshold: float = float(os.getenv("FORESTWATCH_DECREASE", "-0.15"))
    aoi_name: str = "Gorewada Forest, Nagpur, Maharashtra, India"

    def ensure_output_dirs(self):
        for name in ("processed", "features", "metrics", "models", "alerts", "reports"):
            (self.output_path / name).mkdir(parents=True, exist_ok=True)


settings = Settings()
