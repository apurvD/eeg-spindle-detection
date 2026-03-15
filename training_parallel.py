import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count, set_start_method
import os
import time

# Define the Autoencoder model for time series data
class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_length):
        super(TimeSeriesAutoencoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        
        # Encoder: Encodes input into a compressed representation
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten the sequence for dense layers
            nn.Linear(input_size * sequence_length, 128),  # Dense layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Bottleneck layer (smallest representation)
        )
        
        # Decoder: Reconstructs input from the compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size * sequence_length)  # Match the original input size
        )
    
    def forward(self, x):
        # Forward pass: encode the input, then decode back to the original shape
        batch_size = x.shape[0]
        encoded = self.encoder(x)  # Compress input
        decoded = self.decoder(encoded)  # Reconstruct input
        
        # Reshape to (batch, sequence_length, input_size) for output
        decoded = decoded.view(batch_size, self.sequence_length, self.input_size)
        return decoded, encoded

# Dataset class for managing time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.sequence_length = sequence_length
        # Columns containing the time series signals
        self.signal_columns = ['EOG Left', 'EEG C3-A1', 'EEG O1-A1', 'EEG C4-A1', 'EEG O2-A1']
        
        # Convert time components (HH, MM, SS) into a single time column (optional, but could be useful)
        data['time_sec'] = data['HH'].values * 3600 + data['MM'].values * 60 + data['SS'].values
        
        # Standardize the signal columns for training
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data[self.signal_columns])
        
        # Store the data in shared memory (for use with multiprocessing)
        self.shared_data = torch.tensor(self.data, dtype=torch.float32).share_memory_()
        
        # Prepare sliding window indices for creating sequences
        total_sequences = len(self.data) - sequence_length + 1
        self.indices = [(i, i + sequence_length) for i in range(total_sequences)]
            
    def __len__(self):
        # Total number of sequences available
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Fetch a sequence using precomputed indices
        start, end = self.indices[idx]
        return self.shared_data[start:end]

# Function to average gradients across distributed workers
def average_gradients(model):
    size = float(torch.cuda.device_count() if torch.cuda.is_available() else cpu_count())
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data /= size

# Training function executed by each worker
def train_worker(rank, world_size, model, dataset, batch_size, epochs, loss_array, lock, progress_queue):
    # Divide dataset into chunks for each worker
    per_worker = len(dataset) // world_size
    start_idx = rank * per_worker
    end_idx = start_idx + per_worker if rank != world_size - 1 else len(dataset)
    
    print(f"Worker {rank+1}/{world_size} starting with {end_idx - start_idx} samples")
    
    # Subset of the dataset for this worker
    worker_dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
    worker_loader = DataLoader(
        worker_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple threads for data loading
        pin_memory=True
    )
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Reconstruction loss for autoencoder
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in worker_loader:
            optimizer.zero_grad()  # Clear gradients
            reconstructed, _ = model(batch)  # Forward pass
            loss = criterion(reconstructed, batch)  # Compute loss
            loss.backward()  # Backward pass
            
            # Average gradients across all workers
            average_gradients(model)
            
            optimizer.step()  # Update model weights
            epoch_loss += loss.item()
        
        # Update shared loss array with a lock
        with lock:
            loss_array[epoch] += epoch_loss / len(worker_loader)
        
        # Send progress updates to the monitoring queue (only rank 0 reports progress)
        if rank == 0:
            elapsed_time = time.time() - start_time
            progress_queue.put({
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'loss': epoch_loss / len(worker_loader),
                'elapsed_time': elapsed_time
            })
    
    print(f"Worker {rank+1}/{world_size} completed training")

# Function to monitor training progress
def progress_monitor(queue, total_epochs):
    while True:
        progress = queue.get()
        if progress == "DONE":  # Termination signal
            break
        
        # Extract and print progress details
        epoch = progress['epoch']
        loss = progress['loss']
        elapsed_time = progress['elapsed_time']
        
        print(f"Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | Time: {elapsed_time:.1f}s")

# Main function for distributed training
def train_parallel(model, dataset, num_epochs=100, batch_size=32):
    num_workers = min(cpu_count() - 1, 8)  # Limit the number of workers
    if num_workers < 1:
        num_workers = 1  # Ensure there's at least one worker
    
    print(f"\nStarting parallel training with {num_workers} workers")
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}\n")
    
    model.share_memory()  # Share model across processes
    
    # Shared memory for tracking losses
    manager = mp.Manager()
    loss_array = manager.Array('d', [0.0] * num_epochs)
    lock = manager.Lock()
    
    # Start a separate process for monitoring progress
    progress_queue = manager.Queue()
    monitor_process = mp.Process(
        target=progress_monitor,
        args=(progress_queue, num_epochs)
    )
    monitor_process.start()
    
    # Create and start worker processes
    processes = []
    for rank in range(num_workers):
        p = mp.Process(
            target=train_worker,
            args=(rank, num_workers, model, dataset, batch_size, num_epochs, loss_array, lock, progress_queue)
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers to complete
    for p in processes:
        p.join()
    
    # Signal the progress monitor to stop
    progress_queue.put("DONE")
    monitor_process.join()
    
    print("\nTraining completed!")
    return model, list(loss_array)

# Main function to initialize and execute training
def main():
    data_path = 'datasets/final_dataset.csv'
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    sequence_length = 100
    dataset = TimeSeriesDataset(df, sequence_length=sequence_length)
    
    input_size = len(dataset.signal_columns)
    model = TimeSeriesAutoencoder(input_size, sequence_length)
    
    print("\nStarting autoencoder training...")
    model, losses = train_parallel(model, dataset, num_epochs=100, batch_size=32)
    
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': dataset.scaler,
        'sequence_length': sequence_length,
        'input_size': input_size
    }, 'timeseries_autoencoder_par.pth')
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    set_start_method('spawn', force=True)  # Use 'spawn' for multiprocessing
    main()
