from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "geqtrain/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="geqtrain",
    version=version,
    description="GEqTrain is an open-source framework for building, training and deplying E(3)-equivariant graph based models.",
    download_url="https://github.com/limresgrp/GEqTrain",
    author="Daniele Angioletti",
    python_requires=">=3.8",
    packages=find_packages(include=["geqtrain", "geqtrain.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "geqtrain-train = geqtrain.scripts.train:main",
            "geqtrain-evaluate = geqtrain.scripts.evaluate:main",
            "geqtrain-test-equivariance = geqtrain.scripts.test_equivariance:main",
            "geqtrain-deploy = geqtrain.scripts.deploy:main",
        ]
    },
    install_requires=[
        "numpy",
        "einops",
        "tqdm",
        "torch>=1.10.0",
        "wandb>=0.13",
        "e3nn>=0.4.4,<0.6.0",
        "pyyaml",
        "contextlib2;python_version<'3.7'",  # backport of nullcontext
        'contextvars;python_version<"3.7"',  # backport of contextvars for savenload
        "typing_extensions;python_version<'3.8'",  # backport of Final
        "torch-runstats>=0.2.0",
    ],
    zip_safe=True,
)
