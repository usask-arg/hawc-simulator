
(_installation)=
# Installation

The recommended way to install the `hawcsimulator` package is with `uv`.  This guide
assumes that you have installed `uv` by following the [instructions here](https://docs.astral.sh/uv/getting-started/installation/).
It also assumes that you have some knowledge of the command line.

## uv Based Installation

### Python version
If you already have a system Python installation of version >= 3.11 you can skip to the next section.

You can use `uv` to install Python through

    uv python install 3.12

### Create the project
To create a new project, run,

    uv init try-hawcsimulator
    cd try-hawcsimulator

This will have made a new folder in your current directory called `try-hawcsimulator` with a Python
skeleton inside.  To add the `hawcsimulator` package to the project, run,

    uv add hawcsimulator


## Alternative installation
Wheels are made available through `pip`

    pip install hawcsimulator


## Supported Platforms
Currently Python versions >= 3.11 are supported, on Windows(x86), Mac(x86, arm), and Linux (x86, arm) platforms.
