'''
Summary for 3-fold Cross Validation

With Pretrained model:

Fold-1: Accuracy = 0.75073
Fold-2: Accuracy = 0.75389
Fold-3: Accuracy = 0.75120
Avg Accuracy = 0.75194

Without Pretrained model

Fold-1: Accuracy = 0.75073
Fold-2: Accuracy = 0.75072
Fold-3: Accuracy = 0.75072
Avg Accuracy = 0.75072

Performance gain = 0.75194- 0.75072 =0.00122

There is not much performance gain with and without pretrained model only a slight improvement.
I pretrained transformer model for 50 epochs.
I trained full prediction model for 20 epoch in both case (with pretarained and without pretrained model)
I expected the model with pretrained transoformer to perform more better than the model without pretrained
transformer.

'''



import math
import os
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold
import torch.nn.init as init
import gdown
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        self.trans_output=output
        output = self.linear(output)
        return output

def pretrain_tfm_model(model, sequences, epochs,name,device=device):
    model.to(device)
    model.train()
    batch_size = 64
    ntokens=21
    max_length = max(len(seq) for seq in sequences)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    token_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                      'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}

    for epoch in range(epochs):
        total_loss = 0.0
        dataloader = DataLoader(sequences, batch_size=batch_size, shuffle=True)
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            padded_batch = [seq[:max_length] + 'X' * (max_length - len(seq)) for seq in batch]
            encoded_batch = [[token_to_index[token] for token in seq] for seq in padded_batch]
            inputs = torch.LongTensor(encoded_batch).to(device)
            targets = torch.LongTensor([[token_to_index[token] for token in seq[1:]] + [token_to_index['X']] * (max_length - len(seq) + 1) for seq in padded_batch]).to(device)
            output = model(inputs)
            targets=targets.reshape(-1)
            output_flat = output.view(-1, ntokens)
            loss = criterion(output_flat, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), name)


def make_tfm_model(keyword,device=device):
  ntokens = 21  
  emsize = 256  
  d_hid = 256 
  nlayers = 3  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
  nhead = 8  # number of heads in ``nn.MultiheadAttention``
  dropout = 0.2
  model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

  return model

class FullPrediction(nn.Module):

    def __init__(self, antigen, tcr):
        super().__init__()
        self.ant = antigen
        self.tcr = tcr
        self.dense = nn.Linear(256*2, 256)
        self.out_proj = nn.Linear(256, 1)
        self.relu=nn.ReLU()

    def forward(self, antigen_seq, tcr_seq):
        ant = self.ant(antigen_seq)
        emb_ant=self.ant.trans_output
        tcr = self.tcr(tcr_seq)
        emb_tcr=self.tcr.trans_output
        x = torch.cat((emb_ant[:,-1,:], emb_tcr[:,-1,:]), dim=1)
        x = self.dense(x)
        x=self.relu(x)
        x = self.out_proj(x)
        return x
  
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

def make_predict_model(m_ant,m_tcr):
  for params in m_ant.parameters():
    params.requires_grad = False

  for params in m_tcr.parameters():
    params.requires_grad = False
  model=FullPrediction(m_ant,m_tcr)
  return model

def train_model(model, list_antigen_seq, list_tcr_seq, list_interact, epochs, device=device):
    model.to(device)
    model.train()
    batch_size = 64
    max_length_ant = max(len(seq) for seq in list_antigen_seq)
    max_length_tcr = max(len(seq) for seq in list_tcr_seq)
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    token_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                      'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}

    for epoch in range(epochs):
        total_loss = 0.0
        # Create DataLoader for batching
        dataset = list(zip(list_antigen_seq, list_tcr_seq, list_interact))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            # Pad sequences to the maximum length in the batch
            padded_antigens = [seq[:max_length_ant] + 'X' * (max_length_ant - len(seq)) for seq in batch[0]]
            padded_TCRs = [seq[:max_length_tcr] + 'X' * (max_length_tcr - len(seq)) for seq in batch[1]]

            # Convert sequences to token indices
            encoded_antigens = [[token_to_index[token] for token in seq] for seq in padded_antigens]
            encoded_TCRs = [[token_to_index[token] for token in seq] for seq in padded_TCRs]

            # Convert sequences to tensors and move to device
            antigens = torch.LongTensor(encoded_antigens).to(device)
            TCRs = torch.LongTensor(encoded_TCRs).to(device)
            interactions = torch.tensor(batch[2], dtype=torch.float32).to(device)
            outputs = model(antigens,TCRs)
            outputs = outputs.squeeze(1)

            # Calculate loss
            loss = criterion(outputs, interactions)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

def evaluate_model(model, list_antigen_seq, list_tcr_seq, list_interact, device=device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch_size = 64
    max_length_ant = max(len(seq) for seq in list_antigen_seq)
    max_length_tcr = max(len(seq) for seq in list_tcr_seq)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary classification

    token_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                      'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}

    with torch.no_grad():
        dataset = list(zip(list_antigen_seq, list_tcr_seq, list_interact))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        for batch in tqdm(dataloader, desc="Validation"):
            padded_antigens = [seq[:max_length_ant] + 'X' * (max_length_ant - len(seq)) for seq in batch[0]]
            padded_TCRs = [seq[:max_length_tcr] + 'X' * (max_length_tcr - len(seq)) for seq in batch[1]]

            # Convert sequences to token indices
            encoded_antigens = [[token_to_index[token] for token in seq] for seq in padded_antigens]
            encoded_TCRs = [[token_to_index[token] for token in seq] for seq in padded_TCRs]

            # Convert sequences to tensors and move to device
            antigens = torch.LongTensor(encoded_antigens).to(device)
            TCRs = torch.LongTensor(encoded_TCRs).to(device)
            interactions = torch.tensor(batch[2], dtype=torch.float32).to(device)
            outputs = model(antigens,TCRs)
            outputs = outputs.squeeze(1)

            # Calculate loss
            loss = criterion(outputs, interactions)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = (outputs >= 0.5).long()  # Assuming threshold of 0.5 for binary classification
            correct_predictions += (predictions == interactions.long()).sum().item()
            total_samples += len(interactions)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

    return accuracy


def sk_fold_cross_validations(model, list_antigen_seq, list_tcr_seq, list_interact, epochs, k=3, device=device):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    avg_loss = 0.0
    avg_accuracy = 0.0

    for fold, (train_index, val_index) in enumerate(skf.split(list_antigen_seq, list_interact)):
        print(f"Fold {fold + 1}:")

        # Split data into training and validation sets for this fold
        train_antigen_seq = [list_antigen_seq[i] for i in train_index]
        train_tcr_seq = [list_tcr_seq[i] for i in train_index]
        train_interact = [list_interact[i] for i in train_index]

        val_antigen_seq = [list_antigen_seq[i] for i in val_index]
        val_tcr_seq = [list_tcr_seq[i] for i in val_index]
        val_interact = [list_interact[i] for i in val_index]

        # Train the model
        model.reset_parameters()  # Reset model parameters for each fold
        train_model(model, train_antigen_seq, train_tcr_seq, train_interact, epochs, device)

        # Evaluate the model on the validation set
        accuracy = evaluate_model(model, val_antigen_seq, val_tcr_seq, val_interact, device)
        avg_accuracy += accuracy
        print(f"Validation Accuracy: {accuracy}")

        # Save the model
        model_name = f"model_fold_sq{fold + 1}.pth"
        model_path = model_name
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}")

    avg_accuracy /= k
    print(f"Average accuracy: {avg_accuracy}")


def predict(model, L_antigen, L_tcr,device=device):
    predictions = []

    # Define token to index mapping
    token_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                      'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}
    total_iterations = len(L_antigen)
    # Iterate over pairs of antigen and TCR sequences
    for antigen_seq, tcr_seq in tqdm(zip(L_antigen, L_tcr),total=total_iterations, desc="Predicting"):
        # Pad sequences to the maximum length
        max_length_ant = 11#max(len(seq) for seq in L_antigen)
        max_length_tcr = 20#max(len(seq) for seq in L_tcr)
        padded_antigen = antigen_seq + 'X' * (max_length_ant - len(antigen_seq))
        padded_tcr = tcr_seq + 'X' * (max_length_tcr - len(tcr_seq))

        # Convert sequences to token indices
        encoded_antigen = [token_to_index[token] for token in padded_antigen]
        encoded_tcr = [token_to_index[token] for token in padded_tcr]

        # Convert sequences to tensors
        antigen_tensor = torch.LongTensor(encoded_antigen).unsqueeze(0).to(device)
        tcr_tensor = torch.LongTensor(encoded_tcr).unsqueeze(0).to(device)

        # Use the model to make prediction
        with torch.no_grad():
            output = model(antigen_tensor, tcr_tensor)
            prediction = torch.sigmoid(output).item()  # Apply sigmoid activation and get the prediction

            # Convert prediction to 0 or 1
            if prediction >= 0.5:
                prediction = 1
            else:
                prediction = 0

            predictions.append(prediction)

    return predictions

def load_trained_model(model):
    # Define the filename
    fname = 'seq_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Check if the file exists in the current directory
    if os.path.exists(fname):
        try:
            model.load_state_dict(torch.load(fname,map_location=device))
            print(f'Model loaded successfully from {fname}')
            return model
        except Exception as e:
            print(f'Error loading model from {fname}: {e}')
            return None
    else:
        drive_url = 'https://drive.google.com/uc?id=1QvbVuG3x0AzJ2D_OE_I93QNKjFu0MjW_'
        try:
            gdown.download(drive_url, fname, quiet=False)
            model.load_state_dict(torch.load(fname,map_location=device))
            print(f'Model downloaded and loaded successfully from {drive_url}')
            return model
        except Exception as e:
            print(f'Error downloading or loading model from {drive_url}: {e}')
            return None
        


#After calling function from SeqModel run it as following
# import pandas as pd
# import torch
# from SeqModel import pretrain_tfm_model,make_tfm_model,make_predict_model,train_model,evaluate_model,sk_fold_cross_validations,load_trained_model,predict
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# data = pd.read_csv("data.csv")
# antigen_sequences = data['antigen'].tolist()
# tcr_sequences=data['TCR'].tolist()
# interaction=data['interaction'].tolist()
# m_tcr=make_tfm_model('tcr')
# m_ant=make_tfm_model('antigen')

# pretrain_tfm_model(m_tcr, tcr_sequences, 50,f'tcr.pth')
# pretrain_tfm_model(m_ant, antigen_sequences,50,f'antigen.pth')
# m_ant.load_state_dict(torch.load('antigen.pth',map_location=device))
# m_tcr.load_state_dict(torch.load('tcr.pth',map_location=device))

# model=make_predict_model(m_ant,m_tcr)

#sk_fold_cross_validations(model,antigen_sequences,tcr_sequences,interaction,20,3)
# tm=load_trained_model(model)
# pred=predict(tm,antigen_sequences[0:5000],tcr_sequences[0:5000])
# print(pred)

