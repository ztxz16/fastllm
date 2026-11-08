"""Runtime log policy shared by the CLI and persistent-cache startup."""
import os

LOG_LEVEL_ENV = "FASTLLM_PREFIX_CACHE_LOG_LEVEL"
LOG_LEVELS = ("off", "info", "debug")


def prefix_cache_log_level(value=None):
    if value is None:
        value = os.environ.get(LOG_LEVEL_ENV, "info")
    value = str(value).strip().lower()
    if value not in LOG_LEVELS:
        raise ValueError("prefix_cache_log_level must be off, info or debug")
    return value


def configure_prefix_cache_logging(args):
    level = prefix_cache_log_level(getattr(args, "prefix_cache_log_level", None))
    os.environ[LOG_LEVEL_ENV] = level
    return level


def prefix_cache_info(message):
    if prefix_cache_log_level() != "off":
        print(message, flush=True)


def prefix_cache_debug(message):
    if prefix_cache_log_level() == "debug":
        print(message, flush=True)
