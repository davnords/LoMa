import logging
import sys
from logging.handlers import RotatingFileHandler


class _RichFormatter(logging.Formatter):
    """Formatter that injects Rich markup to color the loma package name."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        original_name = record.name
        try:
            if original_name.startswith("loma"):
                parts = original_name.split(".")
                parts[0] = "[bold cyan]" + parts[0] + "[/]"
                record.name = ".".join(parts)
            return super().format(record)
        finally:
            record.name = original_name


logger = logging.getLogger("loma")
logger.propagate = False  # Prevent propagation to avoid double logging
logger.addHandler(logging.NullHandler())  # Default null handler


def configure_logger(
    level=logging.INFO,
    log_format=("%(name)s - %(module)s:%(lineno)d in %(funcName)s - %(message)s"),
    date_format="%Y-%m-%d %H:%M:%S",
    file_path=None,
    file_max_bytes=10485760,  # 10MB
    file_backup_count=3,
    stream=sys.stderr,
    propagate=False,  # Default to False to prevent double logging
    use_rich: bool | None = None,  # Auto-detect by default
):
    """
    Configure the package logger with handlers similar to basicConfig.
    This does NOT use basicConfig() and only affects this package's logger.
    """
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)

    # Set propagation
    logger.propagate = propagate

    # Set level
    logger.setLevel(level)

    # Decide Rich usage
    rich_available = False
    if use_rich is None or use_rich:
        try:
            from rich.logging import RichHandler  # type: ignore

            rich_available = True
        except Exception:
            rich_available = False

    # Create formatters: colored for console, plain for files
    console_formatter = _RichFormatter(log_format)
    file_formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - %(levelname)s - {log_format}", date_format
    )

    # Add console handler if stream is specified
    if stream:
        if rich_available:
            from rich.logging import RichHandler  # type: ignore

            console_handler = RichHandler(
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True,
                markup=True,
                log_time_format=date_format,
                console=None,  # default console
            )
            # Rich will render time/level; our formatter supplies name/caller/message
            console_handler.setFormatter(console_formatter)
        else:
            console_handler = logging.StreamHandler(stream)
            console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Add file handler if file path is specified
    if file_path:
        file_handler = RotatingFileHandler(
            file_path, maxBytes=file_max_bytes, backupCount=file_backup_count
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
