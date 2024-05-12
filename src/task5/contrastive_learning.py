import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn.init as init

import pickle
import numpy as np
from random import randrange
import random
import matplotlib.pyplot as plt
import os

from labels import demo_label_names, eth_label_names, gen_label_names

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def contrastive_loss(emb1, emb2, y, alpha=1):
    if emb1.shape != emb2.shape:
        print(f"The shapes of the embeddings in the contrastive loss are different"
              f"emb1 shape: {emb1.shape}"
              f"emb2 shape: {emb2.shape}"
              f"Exiting...")
        exit()
    
    d = torch.sum((emb1 - emb2)**2, dim=1)
    ls = 1/2*(d)**2 # loss of similar embbedings
    ld = 1/2*(torch.relu(alpha-d))**2 # loss of DISsimilar embbedings
    loss = (1-y)*ls + y*ld
    return loss, ls, ld, d


def plot_training_data(train_loss:list, val_loss:list, path:str):
    # Number of epochs
    epochs = range(1, len(train_loss) + 1)

    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show plot
    plt.savefig(os.path.join(path, 'contrastive_training_stats.png'))


class DenseBottleneckNN(nn.Module):
    def __init__(self, num_classes, bn_dim):
        super(DenseBottleneckNN, self).__init__()
        self.num_classes = num_classes
        self.bn_dim = bn_dim
        self.lin1 =  nn.Linear(2048, 1000)
        self.lin2 =  nn.Linear(1000, 128)
        self.relu = nn.ReLU()
        self.bn = nn.Linear(128, self.bn_dim)
        
        # Initialize weights using Xavier initialization
        init.xavier_uniform_(self.bn.weight)
        init.xavier_uniform_(self.lin1.weight)
        init.xavier_uniform_(self.lin2.weight)
            
    def forward(self, x):
        x = nn.functional.normalize(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        bn = self.bn(x)
        return bn

class ContrastiveLossDataset(Dataset):
    def __init__(self, train:bool, path_to_dataset: str, domain:str = 'labels', train_test_split: float = 0.7) -> None:
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
        
        # with the probability of tossing a coin, select if the data belongs
        # to the same class or to a different one:
        # coin = 0 -> Same class
        # coin = 1 -> Different class
        coin = randrange(2)
        
        if coin:
            # find all the targets that are different from selected item 
            opposite_idxs = np.where(self.targets != self.targets[idx])[0]
            rdm_idx = randrange(len(opposite_idxs))
            rdm_idx = opposite_idxs[rdm_idx]
        else:
            # find targets that are the same class
            same_idxs = np.where(self.targets == self.targets[idx])[0]
            rdm_idx = randrange(len(same_idxs))
            rdm_idx = same_idxs[rdm_idx]
        
        # print(self.targets[idx], self.targets[rdm_idx], coin)
        return self.data[idx], self.data[rdm_idx], coin, self.targets[idx]
    
    
def train(epochs: int, model: DenseBottleneckNN, domain:str, device:str, limit:int=3000):
    # load the model into the device
    model.to(device)
    # load the dataset
    train_ds = ContrastiveLossDataset(train=True,
                                  path_to_dataset=f'../data/embeddings/limit_{limit}/embeddings.pkl',
                                  domain=domain)
    val_ds = ContrastiveLossDataset(train=False,
                                  path_to_dataset=f'../data/embeddings/limit_{limit}/embeddings.pkl',
                                  domain=domain)
    # from the dataset, create the dataloaders
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    optim = Adam(params=model.parameters(), lr=0.025)
    
    # begin the training loop
    loss_train = []
    loss_val = []
    for e in range(epochs):
        epoch_loss = []
        # training step
        model.train()
        for pos, neg, y, _ in train_dl:
            pos, neg, y = pos.to(device), neg.to(device), y.to(device)
            pos_emb = model(pos)
            neg_emb = model(neg)
            
            # loss = custom_triplet_loss(anchor_emb, pos_emb, neg_emb)
            # optimize the network's parameters
            optim.zero_grad()
            # with the embeddings for the final layer computed, 
            # compute the triplet loss.
            loss, _, _, _ = contrastive_loss(pos_emb, neg_emb, y)
            loss = loss.mean()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())
        loss_t = sum(epoch_loss)/len(epoch_loss)
        # validation step
        with torch.no_grad():
            epoch_loss = []
            model.eval()
            for pos, neg, y, _ in val_dl:
                pos, neg, y = pos.to(device), neg.to(device), y.to(device)
                pos_emb = model(pos)
                neg_emb = model(neg)
                # with the embeddings for the final layer computed, 
                # compute the triplet loss.
                loss, _, _, _ = contrastive_loss(pos_emb, neg_emb, y)
                loss = loss.mean()
                # compute the accuracy for the anchor prediction
                epoch_loss.append(loss.item())
            loss_v = sum(epoch_loss)/len(epoch_loss)
        
        if not e % 50 or e == (epochs-1): # print every
            print(f"Epoch {e+1}: Train Loss: {loss_t}, Val Loss: {loss_v}")
            
        loss_train.append(loss_t)
        loss_val.append(loss_v)
            
    return loss_train, loss_val


if __name__ == '__main__':
    epochs = 20
    model_eth = DenseBottleneckNN(num_classes=3, bn_dim=2)
    model_gen = DenseBottleneckNN(num_classes=2, bn_dim=2)
    model_demo = DenseBottleneckNN(num_classes=6, bn_dim=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"INFO: Training using: {device}")
    
    print(f"INFO: Training the demographic model")
    demo_t_loss, demo_v_loss = train(15, model_demo, domain='labels', device=device)
    #print(f"INFO: Training the ethnicity model")
    #eth_t_loss, eth_v_loss = train(750, model_eth, domain='eth_labels', device=device)
    #print(f"INFO: Training the gender model")
    #gen_t_loss, gen_v_loss = train(200, model_gen, domain='gen_labels', device=device)
    
    test_ds = ContrastiveLossDataset(train=True,
                                  path_to_dataset=f'../data/embeddings/limit_1000/embeddings.pkl',
                                  domain='labels')
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
    two_d_embeds = {
        'embeds': [],
        'labels': []
    }
    
    model_demo = model_demo.to(device)
    model_demo.eval()
    with torch.no_grad():
        for anchor, _, _, y in test_dl:
            anchor = anchor.to(device)
            anchor_emb = model_demo(anchor)
            two_d_embeds['embeds'].append(anchor_emb.cpu().numpy())
            two_d_embeds['labels'].append(y.cpu().numpy())
            
    two_d_embeds['embeds'] = np.concatenate(two_d_embeds['embeds'], axis=0)
    two_d_embeds['labels'] = np.concatenate(two_d_embeds['labels'], axis=0)
    
    print(two_d_embeds['embeds'].shape)
    print(two_d_embeds['labels'].shape)
    
    labels = two_d_embeds['labels']
    values = two_d_embeds['embeds']
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(values[indices, 0], values[indices, 1], label=f'{demo_label_names[label]}', alpha=0.5)

    # Add labels and legend
    plt.title('Scatter Plot of Values with Corresponding Labels')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig('cons_2d.png')
    
    plot_training_data(demo_t_loss, demo_v_loss, './task5/demo')