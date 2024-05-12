import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import torch.nn.functional as F
import pickle

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

def plot_training_data(train_data:tuple, val_data:tuple, path:str):
    train_acc, train_loss = train_data
    val_acc, val_loss = val_data
    # Number of epochs
    epochs = range(1, len(train_acc) + 1)

    # Create a subplot with 2 rows and 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting accuracy
    ax1.plot(epochs, train_acc, 'b', label='Training accuracy')
    ax1.plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plotting loss
    ax2.plot(epochs, train_loss, 'b', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show plot
    plt.savefig(os.path.join(path, 'training_stats.png'))
    
    
def accuracy(y_preds, y_true):
    correct_predictions = (y_preds == y_true).sum().item()
    total_predictions = y_true.size(0)
    accuracy = correct_predictions / total_predictions   
    return accuracy 

class EmbedDataset(Dataset):
    def __init__(self, train:bool, path_to_dataset: str, domain:str = 'eth_labels', train_test_split: float = 0.7):
        self.train = train
        self.domain = domain
        
        # Load the dictionary from the pickle file
        with open(path_to_dataset, 'rb') as f:
            dataset = pickle.load(f)
            
        self.embeds = dataset['embeddings']
        self.labels = dataset[domain]
    
        data_len = self.embeds.shape[0]
        split_idx = int(data_len*train_test_split)
                   
        self.data = self.embeds[:split_idx] if self.train else self.embeds[split_idx:]
        self.targets = self.labels[:split_idx] if self.train else self.labels[split_idx:]
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
def train(epochs: int, model: nn.Module, domain:str, device:str, limit:int=50):
    
    model = model.to(device)
    train_ds = EmbedDataset(train=True,
                            path_to_dataset=f'../data/embeddings/limit_{limit}/embeddings.pkl',
                            domain=domain)
    
    val_ds = EmbedDataset(train=False,
                        path_to_dataset=f'../data/embeddings/limit_{limit}/embeddings.pkl',
                        domain=domain)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    if domain == 'eth_labels':
        loss_fn = nn.CrossEntropyLoss()
    elif domain == 'gen_labels':
        loss_fn = nn.BCEWithLogitsLoss()
        
    optim = SGD(model.parameters(), lr=0.025)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    # Training loop
    for e in range(epochs):
        model.train()
        epoch_loss = []
        epoch_acc = []
        for x, y in tqdm(train_dl):
            
            # initializing x and y and forwarding the model 
            probs = -1
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            # depending on the type of problem, initialize the corresponding
            # loss function.
            if type(loss_fn) == nn.CrossEntropyLoss:
                loss = loss_fn(logits, y)
                probs = torch.softmax(logits, dim=0)
            elif type(loss_fn) == nn.BCEWithLogitsLoss:
                y_one_hot = F.one_hot(y, num_classes=model.num_classes)
                loss = loss_fn(logits, y_one_hot.float())
                probs = torch.sigmoid(logits)
            # extracting real prediction and performing backpropagation
            y_pred = torch.argmax(probs, dim=1)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # compute the accuracy and append statistics
            acc = accuracy(y_pred, y)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())
        # aggregate the data using an average
        acc_t = sum(epoch_acc)/len(epoch_acc)
        loss_t = sum(epoch_loss)/len(epoch_loss)
        train_accs.append(acc_t)
        train_losses.append(loss_t)
        
        # after training, we evaluate the performance
        epoch_loss = []
        epoch_acc = []
        with torch.no_grad():
            model.eval()
            for x, y in val_dl:
                
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                if type(loss_fn) == nn.CrossEntropyLoss:
                    val_loss = loss_fn(logits, y)
                    probs = torch.softmax(logits, dim=0)
                elif type(loss_fn) == nn.BCEWithLogitsLoss:
                    y_one_hot = F.one_hot(y).float()
                    val_loss = loss_fn(logits, y_one_hot.float())
                    probs = torch.sigmoid(logits)
                # extracting real prediction, compute the accuracy and append statistics
                y_pred = torch.argmax(probs, dim=1)
                val_acc = accuracy(y_pred, y)
                epoch_acc.append(val_acc)
                epoch_loss.append(val_loss.item())
        acc_v = sum(epoch_acc)/len(epoch_acc)
        loss_v = sum(epoch_loss)/len(epoch_loss)
        val_accs.append(acc_v)
        val_losses.append(loss_v)
              
        print(f"Epoch {e+1}: Train ACC: {acc_t}, Train Loss: {loss_t}, Val ACC: {acc_v}, Val Loss: {loss_v}")
        
    return train_losses, val_losses, train_accs, val_accs


class DenseNN(nn.Module):
    def __init__(self, domain:str='eth_labels'):
        super(DenseNN, self).__init__()
        self.lin1 = nn.Linear(2048, 1000)
        self.relu = nn.ReLU()
        self.num_classes = None
        if domain == 'eth_labels':
            self.num_classes = 3
            self.lin2 = nn.Linear(1000, self.num_classes)
        elif domain == 'gen_labels':
            self.num_classes = 2
            self.lin2 = nn.Linear(1000, self.num_classes)
        else:
            print("The selected domain is wrong, must be either eth_labels or gen_labels.\n")
            exit()
            
    def forward(self, x):
        x = nn.functional.normalize(x)
        return self.lin2(self.relu(self.lin1(x)))

if __name__ == '__main__':
    epochs = 5
    model_eth = DenseNN(domain='eth_labels')
    model_gen = DenseNN(domain='gen_labels')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"INFO: Training using: {device}")
    
    print(f"INFO: Training the ethnicity model")
    eth_t_loss, eth_v_loss, eth_t_acc, eth_v_acc = train(epochs, model_eth, domain='eth_labels', device=device)
    print(f"INFO: Training the gender model")
    gen_t_loss, gen_v_loss, gen_t_acc, gen_v_acc = train(epochs, model_gen, domain='gen_labels', device=device)
    
    #plot_training_data((eth_t_acc, eth_t_loss), (eth_v_acc, eth_v_loss), './eth')
    #plot_training_data((gen_t_acc, gen_t_loss), (gen_v_acc, gen_v_loss), './gen')
    # plot_training_data((gen_t_acc, gen_t_loss), (gen_v_acc, gen_v_loss), './gen')