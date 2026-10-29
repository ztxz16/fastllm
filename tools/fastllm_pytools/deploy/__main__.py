import argparse
from . import add_deploy_args, main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage independent FastLLM replicas')
    add_deploy_args(parser)
    try:
        raise SystemExit(main(parser.parse_args()))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'deploy: {error}\n')
