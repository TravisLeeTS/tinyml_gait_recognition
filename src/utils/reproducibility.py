from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and TensorFlow when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        return


def configure_tensorflow_device(preferred: str = "cpu") -> str:
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if preferred == "gpu" and gpus:
            return "gpu"
        if preferred == "cpu":
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
    except Exception:
        pass
    return "cpu"
