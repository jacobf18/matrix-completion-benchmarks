from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class DatasetBundle:
    dataset_dir: Path
    observed_path: Path
    ground_truth_path: Path
    eval_mask_path: Path

