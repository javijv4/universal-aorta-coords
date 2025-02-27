# universal-aorta-coords

These scripts uses FEniCSx to solve Laplace problems to find the coordinates.
To install the codes and FEniCSx do, 
1. Create a conda environment: `conda create --name uac-env`
2. Activate the environment: `conda activate uac-env`
3. Install FEniCSx: `conda install -c conda-forge fenics-dolfinx mpich h5py cffi python`
4. Install bvmodelgen2, do `python -m pip install -e .` inside the repository folder (make sure the environment is activated).