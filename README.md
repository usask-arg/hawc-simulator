# HAWCsat Simulator Package

[![Documentation Status](https://readthedocs.org/projects/hawc-simulator/badge/?version=latest)](https://usask_arg_example.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/usask-arg/hawc-simulator/main.svg)](https://results.pre-commit.ci/latest/github/usask-arg/usask_arg_example/main)

## Installation
The package can be installed through

`pip install hawcsimulator`

## Usage
Documentation can be found at  https://hawc-simulator.readthedocs.io/

## Development
The development environment is managed with [uv](https://docs.astral.sh/uv/).  By default `uv sync`
installs `skretrieval`, `showlib`, and `ali-processing` from local clones next to this repository;
use `uv sync --no-sources` to use the released versions instead.  See the
[local setup guide](https://hawc-simulator.readthedocs.io/en/latest/developer/local_setup.html) for details.

```
uv sync                       # create .venv with hawcsimulator and the dev tools
uv run pytest                 # run the tests
uv run pre-commit run -a      # lint and format
uv run sphinx-build -b html docs/source docs/build   # build the docs
```

## License
This project is licensed under the MIT license
