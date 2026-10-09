(_dev_local_setup)=
# Local Development Setup

The development environment is managed with [`uv`](https://docs.astral.sh/uv/).
This guide assumes you have `uv` and `git` installed, and have a terminal open where you can access both.

## Cloning the repositories
The `hawc-simulator` package closely depends upon several companion packages, these are,

- `skretrieval`: The core retrieval algorithms used for L1b->L2 processing
- `showlib`: SHOW instrument specific algorithms
- `ali-processing`: ALI instrument specific algorithms
- `sasktran2`: The core radiative transfer model

For development, `hawc-simulator` installs `skretrieval`, `showlib`, and `ali-processing` as editable
installs from sibling folders (see `[tool.uv.sources]` in `pyproject.toml`).  This means they have to be
cloned next to `hawc-simulator`.  We recommend creating a new folder to store all of the repositories,
i.e. `hawc`, and from inside that folder running

```{code}
git clone https://github.com/usask-arg/hawc-simulator
git clone https://github.com/usask-arg/skretrieval
git clone https://github.com/usask-arg/show-lib
git clone https://github.com/usask-arg/ali-processing
```

## Creating the environment
From inside the `hawc-simulator` folder run

```{code}
uv sync
```

This creates a local Python environment in `.venv` containing `hawc-simulator`, the three companion packages
(installed from your local clones, so any changes you make to them are picked up immediately), and the
development tools.  For details on how to use this environment in your preferred IDE see the
[uv documentation](https://docs.astral.sh/uv/).

If you only want to work on `hawc-simulator` itself, you can skip cloning the companion packages and
use the released versions from PyPI instead,

```{code}
uv sync --no-sources
```

## Common commands

```{code}
uv run pytest                 # run the tests
uv run pre-commit run -a      # lint and format
uv run sphinx-build -b html docs/source docs/build   # build the docs
uv run jupyter notebook       # start a notebook server inside the environment
```

## Developing inside sasktran2
`sasktran2` is installed from PyPI by default.  To temporarily use a local clone instead, run

```{code}
uv pip install -e ../sasktran2
```

Note that `sasktran2` contains compiled code, so this requires a working build toolchain (see the
`sasktran2` documentation).  The next `uv sync` will reset the environment back to the released version.
