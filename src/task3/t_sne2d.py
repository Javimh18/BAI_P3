#TASK 2: Using t-SNE, represent the embeddings and its demographic group. Can you differenciate the different demographic groups?
# Class mapping:
#   HA4K_120 -> 0
#   HB4K_120 -> 1
#   HN4K_120 -> 2
#   MA4K_120 -> 3
#   MB4K_120 -> 4
#   MN4K_120 -> 5

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pickle
import os
from labels import demo_label_names, eth_label_names, gen_label_names

if __name__ == '__main__':
    
    limit = 50
    with open(os.path.join(f"../data/embeddings/limit_{limit}/embeddings.pkl"), 'rb') as f:
        dataset = pickle.load(f)
    # loading the embeddings and its labels
    embeddings = dataset['embeddings']
    labels = dataset['labels']
    eth_labels = dataset['eth_labels']
    gen_labels = dataset['gen_labels']
    
    # extract the reduced ethnicity embeddings from the TSNE
    embeddings_reduced = TSNE(n_components=2).fit_transform(embeddings)
    
    df_emb = pd.DataFrame()
    df_emb['x1'] = embeddings_reduced[:, 0]
    df_emb['x2'] = embeddings_reduced[:, 1]
    df_emb['demo_labels'] = [demo_label_names[label] for label in labels]
    df_emb['eth_labels'] = [eth_label_names[label] for label in eth_labels]
    df_emb['gen_labels'] = [gen_label_names[label] for label in gen_labels]
    
    plt.figure(figsize=(16,7))
    ax1 = plt.subplot(1, 3, 1)
    sns.scatterplot(
        x="x1", y="x2",
        hue="demo_labels",
        data=df_emb,
        legend="full",
        alpha=0.9,
        palette=sns.color_palette('Set1'),
        ax=ax1
    )

    ax2 = plt.subplot(1, 3, 2)
    sns.scatterplot(
        x="x1", y="x2",
        hue="eth_labels",
        data=df_emb,
        legend="full",
        alpha=0.9,
        ax=ax2
    )
    
    ax3 = plt.subplot(1, 3, 3)
    sns.scatterplot(
        x="x1", y="x2",
        hue="gen_labels",
        data=df_emb,
        legend="full",
        alpha=0.9,
        ax=ax3
    )
    
    plt.savefig(f"task3/tsne2d_limit{limit}.png")
    
    
    
    