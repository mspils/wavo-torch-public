# wavo-torch

## Usage

Currently only main_hyper.py and predict.py work. Not sure about the Dockerfile
The first for training models and doing hyperparameter optimization, the second for inference.
For the hyperparameter optimization you may need to modify the source code to change the variable storage_base, where optuna logs the experiments to. And create the parent folder.

Without CUDA the whole thing will be VERY slow.

### Example usage:
python main_hyper.py ../data/Halstenbek.csv WHalstenbek_pegel_cm ../models/halstenbek/ 50 --expname experiment_1 --storagename database_01 --pruning
python src/predict.py --config configs/example_torch_single.ini single


## Setup:
git clone 
cd wavo-torch
conda create --name wavo python=3.8
conda activate wavo
pip install -r requirements.txt