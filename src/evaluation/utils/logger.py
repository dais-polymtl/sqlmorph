import inspect
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


class Logger:
    def __init__(self, name=__name__, level="DEBUG"):
        logger.remove()

        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True,
        )
        self.logger = logger.bind(name=name)
        self.name = name

        # Store module name for log directory - handle __main__ special case
        if name == "__main__":
            # Get the actual file name without extension
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            if module:
                self.module_name = os.path.splitext(os.path.basename(module.__file__))[0]
            else:
                self.module_name = "main"
        else:
            # Use the last part of the module name (after the last dot)
            self.module_name = name.split('.')[-1]

    def log(self, level, action, details=None):
        message = {"action": action}
        if details:
            # Make details JSON serializable
            serializable_details = make_json_serializable(details)
            message.update(serializable_details)
        message_str = json.dumps(message)

        frame_info = inspect.stack()[1]
        module = inspect.getmodule(frame_info[0])
        module_name = module.__name__ if module else "unknown"

        if level == "debug":
            self.logger.opt(depth=1).bind(module=module_name).debug(message_str)
        elif level == "info":
            self.logger.opt(depth=1).bind(module=module_name).info(message_str)
        elif level == "warning":
            self.logger.opt(depth=1).bind(module=module_name).warning(message_str)
        elif level == "error":
            self.logger.opt(depth=1).bind(module=module_name).error(message_str)
        elif level == "critical":
            self.logger.opt(depth=1).bind(module=module_name).critical(message_str)

    def get_log_filename(self, prefix=""):
        """
        Creates a log filename with proper directory structure.
        Ensures the directory exists.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory structure using the module name
        log_dir = os.path.join("data", f"{self.module_name}_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create filename
        if prefix:
            filename = f"{prefix}_{timestamp}.json"
        else:
            filename = f"log_{timestamp}.json"

        return os.path.join(log_dir, filename)


def make_json_serializable(obj):
    """
    Convert objects to JSON serializable format.
    Handles NumPy types, Pandas objects, and nested structures.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return list(make_json_serializable(item) for item in obj)
    else:
        return obj