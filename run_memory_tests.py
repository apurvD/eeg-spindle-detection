import torch
import gc
import multiprocessing as mp
import pandas as pd
import psutil
import os
from training_parallel import TimeSeriesDataset, TimeSeriesAutoencoder, train_parallel
from eval_parallel import parallel_evaluation
from torch.multiprocessing import set_start_method

# Utility function to retrieve memory usage
def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size (in MB)
        'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size (in MB)
        'percent': process.memory_percent(),
        'system_percent': psutil.virtual_memory().percent
    }

# Print memory statistics at a given stage
def print_memory_stats(stage):
    """Display memory usage details for a specific stage of execution."""
    mem = get_memory_usage()
    print(f"\nMemory Usage at {stage}:")
    print(f"RSS (Resident Set Size): {mem['rss']:.2f} MB")
    print(f"VMS (Virtual Memory Size): {mem['vms']:.2f} MB")
    print(f"Process Memory Usage: {mem['percent']:.2f}%")
    print(f"System Memory Usage: {mem['system_percent']:.2f}%")

# Perform memory cleanup
def safe_memory_cleanup():
    """Run garbage collection and clear unused memory to optimize resource usage."""
    print_memory_stats("Before Cleanup")
    
    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_memory_stats("After Cleanup")
    print("\nMemory cleaned up safely.")

# Terminate and clean up processes
def cleanup_processes(processes):
    """Terminate all active processes and clean up resources."""
    print_memory_stats("Before Process Cleanup")
    
    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join()
    
    safe_memory_cleanup()
    print("\nProcesses terminated and cleaned up safely.")
    print_memory_stats("After Process Cleanup")

# Initialize dataset and model components
def initialize_components(data_path, sequence_length=100):
    """
    Load dataset and initialize model components for processing.
    This function handles memory tracking during initialization.
    """
    print("\nLoading and initializing components...")
    print_memory_stats("Before Initialization")
    
    try:
        # Load the dataset from a CSV file
        df = pd.read_csv(data_path)
        print_memory_stats("After Loading Data")
        
        # Create a TimeSeriesDataset instance
        dataset = TimeSeriesDataset(df, sequence_length=sequence_length)
        print_memory_stats("After Creating Dataset")
        
        # Initialize the model
        input_size = len(dataset.signal_columns)
        model = TimeSeriesAutoencoder(input_size, sequence_length).cpu()
        
        print_memory_stats("After Model Initialization")
        return dataset, model
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

# Training phase with memory management
def train_with_memory_management(model, dataset, num_epochs=100, batch_size=32):
    """
    Perform the training phase with memory usage tracking.
    """
    try:
        print("\nStarting training phase on CPU...")
        print_memory_stats("Before Training")
        
        # Train the model using the train_parallel function
        trained_model, losses = train_parallel(model, dataset, num_epochs=num_epochs, batch_size=batch_size)
        print_memory_stats("After Training")
        
        safe_memory_cleanup()
        print("Training completed.")
        return trained_model, losses
    except Exception as e:
        print(f"An error occurred during training: {e}")
        raise

# Evaluation phase with memory management
def evaluate_with_memory_management(model_path, dataset):
    """
    Evaluate the trained model in parallel while monitoring memory usage.
    """
    try:
        print("\nStarting evaluation phase on CPU...")
        print_memory_stats("Before Evaluation")
        
        # Use half of the available CPU cores for parallel evaluation
        num_processes = max(1, mp.cpu_count() // 2)
        print(f"Using {num_processes} CPU cores for evaluation...")
        
        # Run parallel evaluation
        average_loss = parallel_evaluation(model_path, dataset, num_processes)
        print_memory_stats("After Evaluation")
        
        safe_memory_cleanup()
        print("Evaluation completed.")
        return average_loss
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        raise

if __name__ == "__main__":
    try:
        print("\nInitial Memory State:")
        print_memory_stats("Startup")
        
        # Set multiprocessing start method
        try:
            set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Ignore if already set
        
        # Configuration details
        data_path = 'datasets/final_dataset.csv'
        model_path = 'models/timeseries_autoencoder.pth'
        num_epochs = 1
        batch_size = 32
        sequence_length = 100

        print("\nStarting CPU-based processing...")
        
        # Initialize components (dataset and model)
        dataset, model = initialize_components(data_path, sequence_length)

        # Perform the training phase
        trained_model, losses = train_with_memory_management(model, dataset, num_epochs, batch_size)

        # Perform the evaluation phase
        if dataset is not None:
            average_loss = evaluate_with_memory_management(model_path, dataset)
            print(f"\nAverage Reconstruction Loss (Parallel): {average_loss}")
            print_memory_stats("Final State")
        else:
            print("Dataset initialization failed.")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
    finally:
        safe_memory_cleanup()
