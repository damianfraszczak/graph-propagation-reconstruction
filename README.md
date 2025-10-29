# Leveraging Structural Information to Reconstruct Hidden Nodes in Propagation Graphs

This repository is based on the [DITTO](https://github.com/q-rz/KDD23-DITTO) method repository and has been extended with additional methods developed as part of the accompanying article. It includes Docker configurations and execution scripts, allowing the solutions to be easily deployed and run in any environment, including setups with CUDA support.

## Dependencies

- CUDA 11.x
- `torch==1.7.1`
- `class-resolver==0.3.10`
- `torch-scatter==2.0.7`
- `torch-sparse==0.6.9`
- `torch-cluster==1.5.9`
- `torch-geometric==2.0.4`
- `ndlib==5.1.1`
- `geopy==2.1.0`

## Usage

To reproduce the results, please run the docker-compose by running `docker-compose up`. By default it will run all scripts with CUDA support.

## Structure
In scripts directory you will find code responsible for running different experiments.
- DITTO - implementation from [DITTO](https://github.com/q-rz/KDD23-DITTO)
- DHREC - implementation from [DITTO](https://github.com/q-rz/KDD23-DITTO)
- CRI - implementation from [DITTO](https://github.com/q-rz/KDD23-DITTO)
- GCN - implementation from [DITTO](https://github.com/q-rz/KDD23-DITTO)
- GIN - implementation from [DITTO](https://github.com/q-rz/KDD23-DITTO)
- SHNI - the method proposed by this paper
- SbRP - paper did not publish their source code, so th has been implemented according to their paper
```
