# HyperPGNN 

Codes for "Path-based Link Prediction on Hyper-relational Knowledge Graph", which has been accepted by IEEE CAI 2024.

## installation

Firstly, you need to install PyTorch and CUDA. After that, you can proceed to PyG. The running environment is a Linux server with Ubuntu and an NVIDIA GeForce RTX 3090 GPU. The CUDA version is 11.3.1.

## Running scripts

Use the following codes to reproduce the results for WD50k. 
```
python script/run.py -c config/WD50k/wd50k.yaml
```
Replace the config path to run experiments on other datasets.

## License
HyperPGNN is released under the MIT license.
