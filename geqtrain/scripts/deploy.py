# geqtrain/scripts/deploy.py

import argparse
import logging

from geqtrain.utils._global_options import set_global_options
from geqtrain.utils.deploy import build_deployment, get_base_deploy_parser
from geqtrain.utils import Config

def main():
    parser = argparse.ArgumentParser(description="Deploy a GEqTrain model.")
    parser.add_argument("--verbose", default="INFO", type=str)
    # Get all the common arguments
    parser = get_base_deploy_parser(parser)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.verbose.upper()))

    model_path = args.model
    config_path = model_path.parent / "config.yaml"
    config = Config.from_file(str(config_path))
    set_global_options(config, warn_on_override=False)

    # Handle the generic --extra-metadata arg
    cli_metadata = {k: v for k, v in (item.split('=') for item in args.extra_metadata)}

    # Call the core build function
    build_deployment(
        model_path=model_path,
        out_file=args.out_file,
        config=config,
        extra_metadata=cli_metadata
    )

if __name__ == "__main__":
    main()