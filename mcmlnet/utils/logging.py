"""Logging utilities."""

import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter for logging."""

    def __init__(self, fmt: str, datefmt: str | None = None):
        """Initialize the custom formatter."""
        super().__init__(fmt, datefmt)

        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.formats = {
            logging.DEBUG: grey + fmt + reset,
            logging.INFO: grey + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        log_fmt = self.formats.get(record.levelno)
        return logging.Formatter(log_fmt).format(record)


def configure_all_loggers(
    numeric_level: int = logging.WARNING, handler: logging.Handler | None = None
) -> None:
    """Configure all existing loggers to use the specified level.

    Args:
        numeric_level: The logging level to set for all loggers.
        handler: Optional handler to add to all loggers. If None, only sets the level.
    """
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)

        if handler is not None:
            # Remove any existing handlers to avoid duplicate output
            for existing_handler in logger.handlers[:]:
                logger.removeHandler(existing_handler)
            # Add the provided handler
            logger.addHandler(handler)
            # Prevent propagation to avoid duplicate messages
            logger.propagate = False


def setup_logging(
    level: str = "warning",
    format: str | None = None,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    context: str = "development",
    use_colors: bool = True,
    logger_name: str | None = None,
) -> logging.Logger:
    """Configure logging with the specified level and format.

    Args:
        level: Logging level name
            ('critical', 'error', 'warning', 'info', 'debug').
        format: Logging format string. If None, uses default format.
        datefmt: Date format string.
        context: The context of the logging.
        use_colors: Whether to use color in the output.
        logger_name: The name of the logger to configure.
            If None, configures the root logger.
    """

    if logger_name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(logger_name)

    level_map = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    numeric_level = level_map.get(level.lower(), logging.WARNING)

    format_templates = {
        "default": "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        "development": "%(asctime)s | %(filename)s:%(lineno)d | "
        "%(levelname)s | %(message)s",
        "minimal": "%(levelname)-8s | %(message)s",
    }

    # Use default format if none provided
    if format is None:
        format = format_templates.get(context, format_templates["development"])

    #  Configure logging
    if use_colors:
        # Use custom colored formatter
        handler = logging.StreamHandler()
        colored_formatter = CustomFormatter(format, datefmt)
        handler.setFormatter(colored_formatter)

        # Get the root logger and add the handler
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Remove existing handlers to avoid duplicates
        for existing_handler in root_logger.handlers[:]:
            root_logger.removeHandler(existing_handler)

        root_logger.addHandler(handler)
    else:
        # Standard logging without colors
        logging.basicConfig(
            format=format,
            level=numeric_level,
            datefmt=datefmt,
        )

    return logger
