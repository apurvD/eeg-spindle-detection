# Virginia Tech- ECE 5510: Multiprocessor Programming Project: 
#@ Concurrent Neural Signal Processor for EEG Spindle Detection
### Submitted by Apurv Sanjay Deshpande (apurvsd@vt.edu) and Hari Makarand Sumant (harisumant@vt.edu) 

## Overview
This project focuses on detecting EEG sleep spindles using neural networks and improving the training and evaluation times through parallel processing techniques. The approach includes preprocessing EEG and EOG datasets, training a Time-Series Autoencoder, and evaluating performance using both sequential and parallel implementations.

## Project Structure
The project directory is structured as follows:


### Python files: 
1. EDA and Preprocessing.ipynb # Jupyter notebook for dataset exploration and preprocessing
2. eval_bench.py # Benchmark evaluation for sequential and parallel methods
3. eval_parallel.py # Evaluation script for parallel implementation
4. eval_sequential.py # Evaluation script for sequential implementation
5. run_memory_tests.py # Script for monitoring memory usage during execution
6. training_bench.py # Benchmark training for sequential and parallel methods
7. training_parallel.py # Parallel training implementation
8. training_sequential.py # Sequential training implementation

- datasets/

1. extrait_wSleepPage01.csv # Raw spindle dataset 1
2. final_dataset.csv # Final combined and preprocessed dataset
3. spindles.csv # Raw spindle dataset 2

- models/

1. timeseries_autoencoder_par.pth # Trained model from parallel execution
2. timeseries_autoencoder.pth # Trained model from sequential execution

- plots/
Contains plots obtained by running all the benchmarks

## Requirements
- Python 3.8+
- Libraries: 
  - `torch`
  - `pandas`
  - `scikit-learn`
  - `psutil`


## How to Run
1. **Preprocessing:**
   - Use `EDA and Preprocessing.ipynb` to preprocess the datasets.

2. **Training:**
   - Run `training_sequential.py` for sequential training.
   - Run `training_parallel.py` for parallel training.

3. **Evaluation:**
   - Use `eval_sequential.py` or `eval_parallel.py` to evaluate the models.

4. **Memory Testing:**
   - Run `run_memory_tests.py` to monitor memory usage during execution.

5. **Benchmarking:**
   - Use `training_bench.py` and `eval_bench.py` to benchmark sequential vs. parallel performance.

