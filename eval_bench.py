import torch
from torch import nn
from torch.utils.data import DataLoader
from training_sequential import TimeSeriesDataset, TimeSeriesAutoencoder
import pandas as pd
import numpy as np
import time
import torch.multiprocessing as mp
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Measure throughput for sequential model evaluation
def measure_sequential_throughput(model, dataset, batch_size=32):
    """Measure the processing throughput of the model in a sequential setup."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()  # Use MSE loss to evaluate reconstruction error
    total_samples = 0
    start_time = time.time()  # Start the timer
    
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # No gradients required during evaluation
        for data in dataloader:
            inputs = data
            if not isinstance(inputs, torch.Tensor):  # Convert to tensor if necessary
                inputs = torch.tensor(inputs, dtype=torch.float32)
            
            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):  # Handle cases where model outputs are tuples
                outputs = outputs[0]
            
            _ = criterion(outputs, inputs)  # Compute loss (not used further)
            total_samples += inputs.size(0)
    
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    throughput = total_samples / elapsed_time  # Calculate throughput
    
    return {
        'total_samples': total_samples,
        'elapsed_time': elapsed_time,
        'throughput': throughput
    }

# Evaluate a single partition of the dataset
def evaluate_partition(rank, model_path, dataset, batch_size, start_event, result_queue):
    """Evaluate a partition of the dataset to measure throughput in parallel."""
    # Synchronize processes by waiting for the start signal
    start_event.wait()
    
    # Load the model and its state from the saved checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model = TimeSeriesAutoencoder(checkpoint['input_size'], checkpoint['sequence_length'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()  # Loss function (not used to backpropagate here)
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data in dataloader:
            inputs = data
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            _ = criterion(outputs, inputs)  # Compute loss
            total_samples += inputs.size(0)
    
    end_time = time.time()
    result_queue.put({
        'rank': rank,
        'total_samples': total_samples,
        'elapsed_time': end_time - start_time
    })

# Measure throughput in parallel evaluation
def measure_parallel_throughput(model_path, dataset, num_processes=None, batch_size=32):
    """Measure the throughput of the model in a parallelized evaluation setup."""
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() // 2)  # Default: half of available CPUs
    
    dataset_size = len(dataset)
    partition_size = dataset_size // num_processes
    partitions = [
        Subset(dataset, 
               range(i * partition_size, 
                     min((i + 1) * partition_size, dataset_size)))
        for i in range(num_processes)
    ]
    
    try:
        mp.set_start_method('fork')  # Use 'fork' for process creation
    except RuntimeError:
        pass
    
    result_queue = mp.Queue()
    start_event = mp.Event()  # Event to synchronize the start
    processes = []
    
    # Create worker processes
    for rank in range(num_processes):
        p = mp.Process(
            target=evaluate_partition,
            args=(rank, model_path, partitions[rank], batch_size, start_event, result_queue)
        )
        p.start()
        processes.append(p)
    
    # Start the evaluation
    time.sleep(0.1)  # Small delay to ensure processes are ready
    start_event.set()
    
    # Collect results from all processes
    results = []
    for _ in range(num_processes):
        results.append(result_queue.get())
    
    for p in processes:
        p.join()
    
    # Use the maximum elapsed time among processes to calculate throughput
    total_time = max(r['elapsed_time'] for r in results)
    total_samples = sum(r['total_samples'] for r in results)
    throughput = total_samples / total_time
    
    return {
        'total_samples': total_samples,
        'elapsed_time': total_time,
        'throughput': throughput,
        'num_processes': num_processes,
        'process_results': results
    }

# Plot throughput comparison
def plot_throughput_comparison(results):
    """Plot a comparison of sequential and parallel throughput."""
    batch_sizes = [r['batch_size'] for r in results]
    seq_throughput = [r['sequential']['throughput'] for r in results]
    par_throughput = [r['parallel']['throughput'] for r in results]
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(batch_sizes))
    
    # Create bar plots
    bars1 = plt.bar(x - width/2, seq_throughput, width, label='Sequential', color='skyblue')
    bars2 = plt.bar(x + width/2, par_throughput, width, label='Parallel', color='lightcoral')
    
    # Annotate the bars with their values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/second)')
    plt.title('Throughput Comparison: Sequential vs Parallel')
    plt.xticks(x, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot speedup achieved through parallel processing
def plot_speedup(results):
    """Plot the speedup factor as a function of batch size."""
    batch_sizes = [r['batch_size'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # Create a line plot
    line = plt.plot(batch_sizes, speedups, marker='o', linewidth=2, color='darkgreen')[0]
    
    # Annotate the speedup values
    for i, speedup in enumerate(speedups):
        plt.text(batch_sizes[i], speedup, f'{speedup:.2f}x',
                ha='center', va='bottom')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup Factor')
    plt.title('Parallel Speedup vs Batch Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('speedup_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all plots for throughput and speedup
def plot_results(results):
    plot_throughput_comparison(results)
    plot_speedup(results)
    print("\nPlots saved: 'throughput_comparison.png' and 'speedup_plot.png'")

# Main function to run evaluations and plot results
def main():
    data_path = 'datasets/final_dataset.csv'
    model_path = 'models/timeseries_autoencoder.pth'
    
    print("Loading data and model...")
    df = pd.read_csv(data_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    sequence_length = checkpoint['sequence_length']
    dataset = TimeSeriesDataset(df, sequence_length=sequence_length)
    
    # Prepare the model for evaluation
    input_size = len(dataset.signal_columns)
    model = TimeSeriesAutoencoder(input_size, sequence_length)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Benchmark settings
    batch_sizes = [32, 64, 128, 256]
    num_processes = max(1, 16)
    
    print("\nRunning throughput benchmarks...")
    results = []
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        seq_results = measure_sequential_throughput(model, dataset, batch_size)
        print(f"Sequential Throughput: {seq_results['throughput']:.2f} samples/second")
        
        par_results = measure_parallel_throughput(model_path, dataset, num_processes, batch_size)
        print(f"Parallel Throughput: {par_results['throughput']:.2f} samples/second")
        
        speedup = par_results['throughput'] / seq_results['throughput']
        print(f"Speedup: {speedup:.2f}x")
        
        results.append({
            'batch_size': batch_size,
            'sequential': seq_results,
            'parallel': par_results,
            'speedup': speedup
        })
    
    print("\nSummary:")
    print("Batch Size | Sequential (samples/s) | Parallel (samples/s) | Speedup")
    print("-" * 65)
    for r in results:
        print(f"{r['batch_size']:^10d} | {r['sequential']['throughput']:^20.2f} | "
              f"{r['parallel']['throughput']:^18.2f} | {r['speedup']:^7.2f}x")
    
    plot_results(results)

if __name__ == '__main__':
    main()
