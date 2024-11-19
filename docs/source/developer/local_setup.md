(_dev_local_setup)=
# Local Development Setup

The preferred way to setup your environment for local development is through [`pixi`](https://pixi.sh/latest/)
this guide will assume you have `pixi` and `git` installed, and have a terminal open where you can access both.

We recommend creating a new folder to store all of the required repositories for the simulator, i.e. `hawc`.

From inside the `hawc` directory, clone the following repositories

```{code}
git clone https://github.com/usask-arg/skretrieval
git clone https://github.com/usask-arg/show-lib
git clone https://github.com/usask-arg/ali-processing
git clone https://github.com/usask-arg/hawc-simulator
```

Then do

```{code}
cd hawc-simulator

pixi install
```

This will create a local Python environment that you can use to develop inside.  For details on how to use
this in your preferred IDE see the [pixi documentation](https://pixi.sh/latest/).