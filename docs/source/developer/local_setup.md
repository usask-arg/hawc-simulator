(_dev_local_setup)=
# Local Development Setup

The preferred way to setup your environment for local development is through [`pixi`](https://pixi.sh/latest/)
this guide will assume you have `pixi` and `git` installed, and have a terminal open where you can access both.

The hawc-simulator package can be cloned onto your machine with

```{code}
git clone https://github.com/usask-arg/hawc-simulator
```

Then do

```{code}
cd hawc-simulator

pixi install
```

This will create a local Python environment that you can use to develop inside.  For details on how to use
this in your preferred IDE see the [pixi documentation](https://pixi.sh/latest/).

## Developing Inside Dependencies
The `hawc-simulator` package closely depends upon several companion packages, these are,

- `skretrieval`: The core retrieval algorithms used for L1b->L2 processing
- `showlib`: SHOW instrument specific algorithms 
- `ali-processing`: ALI instrument specific algorithms
- `sasktran2`: The code radiative transfer model

If you want to develop inside any of these packages and test changes to the simulator, we recommend creating
a new folder to store all of the required repositories for the simulator, i.e. `hawc`.

From inside the hawc directory, clone the the hawc simulator repository and any of the dependencies you want to develop inside.

git clone https://github.com/usask-arg/hawc-simulator
git clone https://github.com/usask-arg/skretrieval
git clone https://github.com/usask-arg/show-lib
git clone https://github.com/usask-arg/ali-processing
git clone https://github.com/usask-arg/sasktran2


Continue installing the simulator as normal,

```{code}
cd hawc-simulator

pixi install
```

Then additional commands can be run to change the installed version of each of these packages to your locally cloned version.
For example, to change the version of `skretrieval` used to your own local installation, run

```{code}
pixi run dev-install-skretrieval
```

The full list of provided commands is,

```{code}
pixi run dev-install-skretrieval
pixi run dev-install-showlib
pixi run dev-install-aliprocessing
pixi run dev-install-sasktran2
```

You can run any number of these to change the used version to your local installed version.  To reset
back to the official versions, you can do

```{code}
pixi clean
pixi install
```