## Running the experiments

### Introduction:
This is a patch version of [graph-neural-pde](https://github.com/twitter-research/graph-neural-pde) 
resolve errors while running experiments on local environment. Providing with several visualization notebooks
for visualising networks experiment on the project.


### Requirements
Dependencies (with python >= 3.8):

Main dependencies are
- torch==1.13.0
- torch-cluster==1.6.0
- torch-geometric==2.2.0
- torch-scatter==2.1.0
- torch-sparse==0.6.15
- torch-spline-conv==1.2.1
- torchdiffeq==0.2.3

Commands to install all the dependencies in a new conda environment
```
conda env create -f environment.yml 
conda activate grand

pip install ogb pykeops
pip install torch==1.13.0
pip install torchdiffeq

pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install torch-spline-conv
pip install torch-geometric
```

### Troubleshooting

There is a bug in torch-sparse==0.6.15 occur with Apple Silicon chips. 

Resolve by running:
```pip install git+https://github.com/rusty1s/pytorch_sparse.git```

## GRAND (Graph Neural Diffusion)

### Dataset and Preprocessing
create a root level folder
```
./data
```
This will be automatically populated the first time each experiment is run.

### Experiments
For example to run for Cora with random splits:
```
cd src
python run_GNN.py --dataset Cora 
```

### Visualization
- Attention Visualization is available on `visualise_attention.ipynb`
- MNIST Visualization is available on `mnist_visualise.ipynb`
