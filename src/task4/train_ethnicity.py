# Class mapping:
#   HA4K_120 -> 0
#   HB4K_120 -> 1
#   HN4K_120 -> 2
#   MA4K_120 -> 3
#   MB4K_120 -> 4
#   MN4K_120 -> 5

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

MAX_SAMPLES = 750
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def class_mapping(label: str):
    
    if label == 'asian':
        return 0
    elif label == 'caucasian':
        return 1
    elif label == 'black':
        return 2
    else:
        print("The label is not recognized... exiting...")
        exit()
        
        
def accuracy(y_preds, y_true):
    correct_predictions = (y_preds == y_true).sum().item()
    total_predictions = y_true.size(0)
    accuracy = correct_predictions / total_predictions   
    return accuracy 


class EthnicityDataset(Dataset):
    def __init__(self, train:bool, path_to_dataset:str, domain:str = 'eth_labels', ethnicity: str='asian', train_test_split: float = 0.66):
        self.train = train
        self.domain = domain
        self.ethnicity = ethnicity
        
        with open(path_to_dataset, 'rb') as f:
            dataset = pickle.load(f)
        
        eth_labels = dataset[self.domain]
        gen_labels = dataset['gen_labels']
        
        if ethnicity != 'all':
            target_idx_eth_labels = np.where(eth_labels == class_mapping(ethnicity))[0]
            target_eth_embeddings = dataset['embeddings'][target_idx_eth_labels]
            target_gen_labels = gen_labels[target_idx_eth_labels]
        else:
            target_eth_embeddings = dataset['embeddings']
            target_gen_labels = dataset['gen_labels']
        
        nsamples = target_eth_embeddings.shape[0]
        split_idx = int(train_test_split*nsamples)
        
        self.data = target_eth_embeddings[:split_idx] if self.train else target_eth_embeddings[split_idx:]
        self.targets = target_gen_labels[:split_idx] if self.train else target_gen_labels[split_idx:]
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    
class DenseNN(nn.Module):
    def __init__(self, num_classes):
        super(DenseNN, self).__init__()
        self.num_classes = num_classes
        self.lin1 = nn.Linear(2048, 1000)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(1000, self.num_classes)
            
    def forward(self, x):
        # x = nn.functional.normalize(x)
        return self.lin2(self.relu(self.lin1(x)))


def train(epochs: int, model: nn.Module, domain:str, device:str, ethnicity:str, limit:int=1000):
    
    model = model.to(device)
    train_ds = EthnicityDataset(train=True,
                            path_to_dataset=f'../data/embeddings/limit_{limit}/embeddings.pkl',
                            domain=domain,
                            ethnicity=ethnicity)
    
    val_ds = EthnicityDataset(train=False,
                        path_to_dataset=f'../data/embeddings/limit_{limit}/embeddings.pkl',
                        domain=domain,
                        ethnicity=ethnicity)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    
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
                    y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()
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


def evaluate(model: nn.Module, val_dl:DataLoader, device:str):
    
    loss_fn = nn.BCEWithLogitsLoss()
    epoch_loss = []
    epoch_acc = []
    with torch.no_grad():
        model.eval()
        for x, y in val_dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()
            val_loss = loss_fn(logits, y_one_hot)
            probs = torch.sigmoid(logits)
            # extracting real prediction, compute the accuracy and append statistics
            y_pred = torch.argmax(probs, dim=1)
            val_acc = accuracy(y_pred, y)
            epoch_acc.append(val_acc)
            epoch_loss.append(val_loss.item())
        # accumulate the epoch results
        acc_v = sum(epoch_acc)/len(epoch_acc)
        loss_v = sum(epoch_loss)/len(epoch_loss)
        
    return acc_v, loss_v
    
    
if __name__ == '__main__':
    epochs = 5
    model_asian = DenseNN(num_classes=2)
    model_black = DenseNN(num_classes=2)
    model_caucasian = DenseNN(num_classes=2)
    model_all = DenseNN(num_classes=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"INFO: Training using: {device}")
    
    print(f"INFO: Training the asian model")
    eth_t_loss, eth_v_loss, eth_t_acc, eth_v_acc = train(epochs, model_asian, domain='eth_labels', device=device, ethnicity='asian')
    print(f"INFO: Training the black model")
    gen_t_loss, gen_v_loss, gen_t_acc, gen_v_acc = train(epochs, model_black, domain='eth_labels', device=device, ethnicity='black')
    print(f"INFO: Training the caucasian model")
    gen_t_loss, gen_v_loss, gen_t_acc, gen_v_acc = train(epochs, model_caucasian, domain='eth_labels', device=device, ethnicity='caucasian')
    print(f"INFO: Training the general model")
    gen_t_loss, gen_v_loss, gen_t_acc, gen_v_acc = train(epochs, model_all, domain='eth_labels', device=device, ethnicity='all')
    
    # declare the ethnicities dataset
    asian_val_ds = EthnicityDataset(train=False,
                        path_to_dataset=f'../data/embeddings/limit_500/embeddings.pkl',
                        domain='eth_labels',
                        ethnicity='asian')
    
    black_val_ds = EthnicityDataset(train=False,
                        path_to_dataset=f'../data/embeddings/limit_500/embeddings.pkl',
                        domain='eth_labels',
                        ethnicity='black')
    
    caucasian_val_ds = EthnicityDataset(train=False,
                        path_to_dataset=f'../data/embeddings/limit_500/embeddings.pkl',
                        domain='eth_labels',
                        ethnicity='caucasian')
    
    # declare the dataloaders
    asian_val_dl = DataLoader(asian_val_ds, 32, shuffle=False)
    black_val_dl = DataLoader(black_val_ds, 32, shuffle=False)
    caucasian_val_dl = DataLoader(caucasian_val_ds, 32, shuffle=False)
    
    # store in an array the models and the dataloaders to iterate through them
    models = [model_asian, model_black, model_caucasian, model_all]
    dls = [asian_val_dl, black_val_dl, caucasian_val_dl]
    
    # get the scores
    acc_scores = np.zeros(shape=(len(models), len(dls))) # rows models, columns dataloaders
    for i in range(len(models)):
        for j in range(len(dls)):
            acc_scores[i, j], _ = evaluate(model=models[i],
                                           val_dl=dls[j],
                                           device=device)
    print("Result scores:")
    print(acc_scores)