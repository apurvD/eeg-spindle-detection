import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Subset
from training_sequential import TimeSeriesDataset, TimeSeriesAutoencoder
import pandas as pd
import numpy as np
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Function to evaluate a single partition of the dataset
def evaluate_partition(rank, model_path, dataset, queue):
    """
    Evaluate a partition of the dataset on CPU and compute the total loss.
    Results are pushed into a multiprocessing queue for aggregation.
    """
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model = TimeSeriesAutoencoder(checkpoint['input_size'], checkpoint['sequence_length'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    total_loss = 0
    num_batches = 0
    
    # Create a DataLoader for the partition
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():  # No gradients required during evaluation
        for data in dataloader:
            inputs = data
            if not isinstance(inputs, torch.Tensor):  # Ensure inputs are tensors
                inputs = torch.tensor(inputs, dtype=torch.float32)
            
            outputs = model(inputs)
            if isinstance(outputs, tuple):  # Handle tuple outputs (e.g., decoded + latent vector)
                outputs = outputs[0]
            
            loss = criterion(outputs, inputs)  # Compute loss for this batch
            total_loss += loss.item()
            num_batches += 1
    
    # Push the total loss and number of batches to the queue
    queue.put((total_loss, num_batches))

# Function to run evaluation in parallel across multiple processes
def parallel_evaluation(model_path, dataset, num_processes=None):
    """
    Perform parallel evaluation of the model by splitting the dataset across
    multiple CPU processes.
    """
    if num_processes is None:
        # Default: Use half the available CPU cores for evaluation
        num_processes = max(1, mp.cpu_count() // 2)
    
    # Split the dataset into equal-sized partitions for each process
    dataset_size = len(dataset)
    partition_size = dataset_size // num_processes
    partitions = [
        Subset(dataset, 
               range(i * partition_size, 
                     min((i + 1) * partition_size, dataset_size)))
        for i in range(num_processes)
    ]
    
    # Set multiprocessing method to 'fork' (works efficiently on Unix-like systems)
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass  # Ignore if the start method is already set
    
    # Create a multiprocessing queue to collect results
    queue = mp.Queue()
    
    # Launch processes for evaluation
    processes = []
    for rank in range(num_processes):
        p = mp.Process(
            target=evaluate_partition,
            args=(rank, model_path, partitions[rank], queue)
        )
        p.start()
        processes.append(p)
    
    # Gather results from all processes
    total_loss = 0
    total_batches = 0
    for _ in range(num_processes):
        loss, num_batches = queue.get()
        total_loss += loss
        total_batches += num_batches
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Compute the average reconstruction loss
    average_loss = total_loss / total_batches
    return average_loss

if __name__ == '__main__':
    # Load dataset from CSV file
    df = pd.read_csv('datasets/final_dataset.csv')
    
    # Load model checkpoint to retrieve required parameters
    checkpoint = torch.load(
        'models/timeseries_autoencoder.pth', 
        map_location='cpu'
    )
    sequence_length = checkpoint['sequence_length']  # Get sequence length from the checkpoint
    
    # Create the dataset object
    dataset = TimeSeriesDataset(df, sequence_length=sequence_length)
    
    # Set path to the model checkpoint
    model_path = 'models/timeseries_autoencoder.pth'
    
    # Determine the number of processes to use (default: use all available cores)
    num_processes = max(mp.cpu_count(), 8)
    print(f"Running evaluation with {num_processes} processes...")
    
    # Perform parallel evaluation
    average_loss = parallel_evaluation(model_path, dataset, num_processes)
    print(f"Average Reconstruction Loss (Parallel): {average_loss}")
