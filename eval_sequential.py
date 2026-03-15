import torch
from torch import nn
from torch.utils.data import DataLoader
from training_sequential import TimeSeriesDataset, TimeSeriesAutoencoder  # Replace with your dataset class
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the saved model dictionary
checkpoint = torch.load('models/timeseries_autoencoder.pth')

# Load the dataset scaler and parameters
scaler = checkpoint['scaler']
sequence_length = checkpoint['sequence_length']
input_size = checkpoint['input_size']
# Initialize the model
model = TimeSeriesAutoencoder(input_size, sequence_length)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load the dataset
df = pd.read_csv('datasets/final_dataset.csv')
dataset = TimeSeriesDataset(df, sequence_length=sequence_length)  # Adjust if `scaler.transform` is needed
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Evaluation loop
criterion = nn.MSELoss()
total_loss = 0

with torch.no_grad():
    for data in dataloader:
        # Ensure data is a tensor
        inputs = data
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)

        # Ensure input and output shapes match
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use the primary output

        # print("Input shape:", inputs.shape)
        # print("Output shape:", outputs.shape)

        # Compute loss
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

average_loss = total_loss / len(dataloader)
print(f"Average Reconstruction Loss: {average_loss}")
