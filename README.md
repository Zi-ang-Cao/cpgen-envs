# CP-Gen environments

This repository hosts a collection of tasks using in CP-Gen (Constraint-Preserving Data Generation). It is intended to standardize
experimental validation of simulated tasks and expedite the
development of new environments.

Git clone the repository and then do:

``` shell
pip install -e .
```

Next, `git clone https://github.com/kevin-thankyou-lin/mimicgen.git` and `pip install -e mimicgen`.

Currently, this repo contains the following envs:
- `ThreePieceAssembly` env: adapted from the [mimicgen repo](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/envs/robosuite/three_piece_assembly.py) to work with `robosuite==1.5.0`.
- `NutAssemblySquare` env: main change: `SquareNutObject.bottom_offset = np.array([0, 0, 0.01])` so object doesn't start high and fall.
- `MugCleanup` env: adapted from the [mimicgen repo](https://github.com/NVlabs/mimicgen/blob/main/mimicgen/envs/robosuite/mug_cleanup.py) to work with `robosuite==1.5.0`.

To add a new env, please update `cpgen_envs/__init__.py` by importing the relevant file so that robosuite can "register" the env.

Note: part of the environment source code uses mimicgen's code which has it's own [license](https://github.com/NVlabs/mimicgen?tab=readme-ov-file#license).
