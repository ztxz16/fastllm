"""Compatibility entrypoint for the standalone FastLLM WebUI."""

try:
    from .webui_server import main
except ImportError:
    from webui_server import main


if __name__ == "__main__":
    main()
