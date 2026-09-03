"""Legacy entrypoint kept for users of ``python web_demo.py``."""

try:
    from .webui_server import main
except ImportError:
    from webui_server import main


if __name__ == "__main__":
    main()
