from loguru import logger
import json
import sys
import inspect


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

    def log(self, level, action, details=None):
        message = {"action": action}
        if details:
            message.update(details)
        message_str = json.dumps(message)

        frame_info = inspect.stack()[1]
        module = inspect.getmodule(frame_info[0])
        module_name = module if module else "unknown"

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
