import time
import numpy as np
import matplotlib.pyplot as plt
from training_sequential import TimeSeriesAutoencoder, TimeSeriesDataset, train_autoencoder
from training_parallel import train_parallel
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

# Measure sequential training time
def measure_sequential_time(model, dataset, num_epochs=100, batch_size=32):
    """
    Measure the pure training time for sequential execution, using MPS (if available).
    """
    # Check for MPS (Metal Performance Shaders for macOS) and set the device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for sequential training")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU for sequential training")
    
    model = model.to(device)  # Move the model to the appropriate device
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    pure_training_time = 0  # Keep track of total training time
    
    # Train the model and measure the time per epoch
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
        epoch_end = time.time()
        pure_training_time += epoch_end - epoch_start  # Accumulate epoch time
        
        if (epoch + 1) % 10 == 0:  # Print progress every 10 epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Time: {pure_training_time:.2f}s')
    
    model = model.to('cpu')  # Move the model back to CPU
    return pure_training_time

# Parallel training worker function
def parallel_training_worker(rank, model, dataset_partition, batch_size, num_epochs, training_time_queue, ready_queue, start_event):
    """
    Worker process for parallel training that measures pure training time for its partition.
    """
    data_loader = DataLoader(dataset_partition, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Signal that the worker is ready
    ready_queue.put(rank)
    
    # Wait for the start signal from the main process
    start_event.wait()
    
    pure_training_time = 0  # Measure pure training time for this worker
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for batch in data_loader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
        epoch_end = time.time()
        pure_training_time += epoch_end - epoch_start  # Accumulate time
    
    training_time_queue.put(pure_training_time)  # Send the training time to the main process

# Measure parallel training time
def measure_parallel_time(model, dataset, num_threads, num_epochs=100, batch_size=32):
    """
    Measure the pure training time for parallel execution using multiple threads.
    """
    model.share_memory()  # Allow the model to be shared across processes
    
    # Divide the dataset into partitions for each thread
    dataset_size = len(dataset)
    partition_size = dataset_size // num_threads
    partitions = [
        Subset(dataset, 
               range(i * partition_size, 
                     min((i + 1) * partition_size, dataset_size)))
        for i in range(num_threads)
    ]
    
    # Create communication mechanisms
    training_time_queue = mp.Queue()
    ready_queue = mp.Queue()
    start_event = mp.Event()
    
    # Start worker processes
    processes = []
    for rank in range(num_threads):
        p = mp.Process(
            target=parallel_training_worker,
            args=(rank, model, partitions[rank], batch_size, num_epochs, 
                  training_time_queue, ready_queue, start_event)
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers to be ready
    ready_workers = 0
    while ready_workers < num_threads:
        ready_queue.get()
        ready_workers += 1
    
    # Signal workers to begin training
    start_event.set()
    
    # Collect training times from all workers
    training_times = []
    for _ in range(num_threads):
        training_times.append(training_time_queue.get())
    
    # Ensure all processes complete
    for p in processes:
        p.join()
    
    # Return the maximum training time across all workers
    return max(training_times)

# Main function for measuring and comparing training times
def main():
    print("Loading data...")
    df = pd.read_csv('datasets/final_dataset.csv')
    
    sequence_length = 100
    dataset = TimeSeriesDataset(df, sequence_length=sequence_length)
    input_size = len(dataset.signal_columns)
    thread_configurations = [4, 8, 12]  # Configurations for parallel execution
    
    # Measure sequential training time
    sequential_model = TimeSeriesAutoencoder(input_size, sequence_length)
    print("\nMeasuring sequential training time...")
    sequential_time = measure_sequential_time(sequential_model, dataset)
    print(f"Pure Sequential Training Time: {sequential_time:.2f} seconds")
    
    # Store results for comparison
    results = []
    
    # Test parallel execution with different thread configurations
    for num_threads in thread_configurations:
        print(f"\nTesting with {num_threads} threads...")
        parallel_model = TimeSeriesAutoencoder(input_size, sequence_length)
        
        try:
            parallel_time = measure_parallel_time(parallel_model, dataset, num_threads)
            speedup = sequential_time / parallel_time
            parallel_fraction = (num_threads / speedup - 1) / (num_threads - 1)
            parallel_fraction = max(0.0, min(1.0, parallel_fraction))  # Clamp between 0 and 1
            theoretical_max = 1 / ((1 - parallel_fraction) + parallel_fraction / num_threads)
            
            results.append({
                'num_threads': num_threads,
                'parallel_time': parallel_time,
                'observed_speedup': speedup,
                'parallel_fraction': parallel_fraction,
                'theoretical_max_speedup': theoretical_max
            })
            
            print(f"Pure Parallel Training Time: {parallel_time:.2f} seconds")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Parallel Fraction: {parallel_fraction:.2%}")
            print(f"Theoretical Maximum Speedup: {theoretical_max:.2f}x")
            
        except Exception as e:
            print(f"Error with {num_threads} threads: {str(e)}")
    
    # Save results to a CSV and display a summary
    if results:
        results_df = pd.DataFrame(results)
        results_df['sequential_time'] = sequential_time
        results_df.to_csv('pure_training_times.csv', index=False)
        print("\nResults saved to 'pure_training_times.csv'")
        print("\nSummary:")
        print(results_df.to_string(float_format=lambda x: f"{x:.2f}"))

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Set multiprocessing method to spawn
    main()
