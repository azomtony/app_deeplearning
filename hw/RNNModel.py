import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import gdown
import os

def marker_with_padding(names):
    #maximum length among all names
    max_length = max(len(name) for name in names) + 1  # Maximum length with markers
    
    input_names = []
    output_names = []
    for name in names:
        #markers for beginning and end of the name
        input_name = '^' + name
        output_name = name + '.'
        
        # Padded with spaces
        input_name_padded = input_name.ljust(max_length)
        output_name_padded = output_name.ljust(max_length)
        
        input_names.append(input_name_padded)
        output_names.append(output_name_padded)
    return input_names, output_names

def name_2_vec(name):
  num_name=len(name)
  max_length=max(len(word) for word in name)
  encoded_vec=np.zeros((num_name,max_length,28))
  char_to_idx = {char: idx for idx, char in enumerate('^abcdefghijklmnopqrstuvwxyz.')}
  for i,word in enumerate(name):
    for j,char in enumerate(word):
      if char in char_to_idx:
        index = char_to_idx[char]
        encoded_vec[i, j, index] = 1
  return encoded_vec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2).double()
        self.fc = nn.Linear(hidden_size, output_size).double()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)  # Get output from all hidden states

        return output
    
def create_model(mlen):

  input_size = 28  # Number of possible characters including '^' and '.'
  hidden_size = 128
  output_size = input_size  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = LSTMModel(input_size, hidden_size, output_size).to(device)
  return model
    
def train_model(model,n_epochs):
    with open('yob2018.txt') as file:
      lines=file.readlines()
    names=[line.split(',')[0] for line in lines]
    filtered_name=[name.lower() for name in names if re.match("^[a-zA-Z]+$",name)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    input_name,output_name=marker_with_padding(filtered_name)
    encoded_input=name_2_vec(input_name)
    encoded_output=name_2_vec(output_name)
    input_tensor=torch.tensor(encoded_input,dtype=torch.float64).to(device)
    output_tensor=torch.tensor(encoded_output,dtype=torch.float64).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = TensorDataset(input_tensor, output_tensor)
    train_loader = DataLoader(train_data, shuffle=True)
    epoch_losses = []

    for epoch in range(n_epochs):
        total_loss = 0
        model.train()

        for input_batch, output_batch in train_loader:
            input_batch, output_batch = input_batch.to(device), output_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(input_batch)
            output = output.view(-1, output.size(2))  # Reshape to [batch_size * sequence_length, output_size]
            target = output_batch.view(-1, output_batch.size(2)).argmax(dim=1)  # Reshape and get class indices
            # Create a mask tensor to identify padding positions
            mask = (input_batch.sum(dim=2) != 0).float()  # Check if all characters are zero along the third dimension
            # Calculate loss
            loss = criterion(output, target)  # Calculate individual losses
            # Multiply the loss tensor by the mask tensor to zero out loss values at padding positions
            masked_loss = loss * mask.view(-1)
            # Calculate total loss only over non-padding positions
            num_non_padding = mask.sum()  # Total number of non-padding positions
            total_loss += masked_loss.sum() / num_non_padding  # Sum of individual losses over non-padding positions
            # Backward pass and optimization
            (masked_loss.sum() / num_non_padding).backward()  # Backward pass with masked loss
            optimizer.step()

        # Print average loss for the epoch
        epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}')

    # Plot the training loss curve and save the model
    plt.plot([loss.detach().item() for loss in epoch_losses])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('training_loss_curve.png')
    torch.save(model.state_dict(), 'rnn_model.pth')

    return model

def load_trained_model():
    # Define the filename
    fname = 'rnn_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if the file exists in the current directory
    if os.path.exists(fname):
        try:
            model = create_model(1)
            model.load_state_dict(torch.load(fname,map_location=device))
            print(f'Model loaded successfully from {fname}')
            return model
        except Exception as e:
            print(f'Error loading model from {fname}: {e}')
            return None
    else:
        drive_url = 'https://drive.google.com/uc?id=1W7E_mlELwkcuEAbb3Qx_NRAK2kswZcel'
        try:
            gdown.download(drive_url, fname, quiet=False)
            model = create_model(1)
            model.load_state_dict(torch.load(fname,map_location=device))
            print(f'Model downloaded and loaded successfully from {drive_url}')
            return model
        except Exception as e:
            print(f'Error downloading or loading model from {drive_url}: {e}')
            return None


def generate_name(model, max_length):
    char_to_idx = {char: idx for idx, char in enumerate('^abcdefghijklmnopqrstuvwxyz.')}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    input_tensor = torch.zeros(1, 1, len(char_to_idx), device=device, dtype=model_dtype)
    input_tensor[0, 0, char_to_idx['^']] = 1
    generated_name = ''
    for _ in range(max_length):
        output = model(input_tensor)
        prob_distribution = torch.softmax(output.squeeze(), dim=0).cpu().detach().numpy()
        sampled_char_idx = np.random.multinomial(1, prob_distribution).argmax()
        generated_name += idx_to_char[sampled_char_idx]
        
        if sampled_char_idx == char_to_idx['.']:
            break

        input_tensor = torch.zeros(1, 1, len(char_to_idx), device=device, dtype=model_dtype)
        input_tensor[0, 0, sampled_char_idx] = 1

    if generated_name.endswith('.'):
        generated_name = generated_name[:-1]
    
    return generated_name
