import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define a custom PyTorch Dataset for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.sequence_length = sequence_length
        self.signal_columns = ['EOG Left', 'EEG C3-A1', 'EEG O1-A1', 'EEG C4-A1', 'EEG O2-A1']
        
        # Combine HH, MM, SS columns into a single time feature (in seconds)
        data['time_sec'] = data['HH'] * 3600 + data['MM'] * 60 + data['SS']
        
        # Standardize the selected signal columns
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data[self.signal_columns])
        
        # Generate sequences for time series modeling
        self.sequences = []
        for i in range(len(self.data) - sequence_length + 1):
            sequence = self.data[i:(i + sequence_length)]
            self.sequences.append(sequence)
            
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return the sequence as a PyTorch tensor
        return torch.FloatTensor(self.sequences[idx])

# Define a simple Autoencoder for time series reconstruction
class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_length):
        super(TimeSeriesAutoencoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        
        # Encoder: Compress input into a latent representation
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flatten (sequence_length, input_size) to (sequence_length * input_size)
            nn.Linear(input_size * sequence_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Latent space
        )
        
        # Decoder: Reconstruct input from the latent representation
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size * sequence_length)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Encode input
        encoded = self.encoder(x)
        # Decode back to input dimensions
        decoded = self.decoder(encoded)
        # Reshape to (batch_size, sequence_length, input_size)
        decoded = decoded.view(batch_size, self.sequence_length, self.input_size)
        return decoded, encoded

# Training function for the autoencoder
def train_autoencoder(model, train_loader, num_epochs=100, device='cpu'):
    """
    Train the autoencoder on the given data loader.
    """
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.8f}')

# Function to generate anomaly labels based on reconstruction error
def generate_labels(model, data_loader, threshold_percentile=95, device='cpu'):
    """
    Generate anomaly labels by calculating reconstruction errors and comparing to a threshold.
    """
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            # Calculate reconstruction error (mean squared error)
            error = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
            reconstruction_errors.extend(error.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    # Determine the threshold from the specified percentile
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    # Assign labels (1 for anomaly, 0 for normal)
    labels = (reconstruction_errors > threshold).astype(int)
    
    return labels, reconstruction_errors

# Main function to train the autoencoder and generate labeled data
def main():
    # Check if MPS (Metal Performance Shaders) is available, otherwise use CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load the dataset from CSV
    df = pd.read_csv('datasets/final_dataset.csv')
    
    # Create the dataset and data loader
    sequence_length = 100
    dataset = TimeSeriesDataset(df, sequence_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize the model
    input_size = len(dataset.signal_columns)
    model = TimeSeriesAutoencoder(input_size, sequence_length).to(device)
    
    # Train the autoencoder
    print("Training autoencoder...")
    train_autoencoder(model, data_loader, num_epochs=100, device=device)
    
    # Save the trained model
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': dataset.scaler,
        'sequence_length': sequence_length,
        'input_size': input_size
    }, 'timeseries_autoencoder.pth')
    
    # Generate anomaly labels
    print("\nGenerating labels...")
    labels, errors = generate_labels(model, data_loader, device=device)
    
    # Add labels to the original dataframe
    df_labeled = df.copy()
    pad_length = sequence_length - 1  # Adjust for initial sequences without labels
    labels = np.pad(labels, (pad_length, 0), 'edge')
    df_labeled['Label'] = labels
    
    # Save the labeled dataframe
    df_labeled.to_csv('labeled_output_ml.csv', index=False)
    print("\nLabel distribution:")
    print(pd.Series(labels).value_counts(normalize=True))
    print("\nLabeled data saved to 'labeled_output_ml.csv'")

if __name__ == "__main__":
    main()
