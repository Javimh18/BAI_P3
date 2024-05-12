# TASK 2: Using t-SNE, represent the embeddings and its demographic group. Can you differenciate the different demographic groups?
# Class mapping:
#   HA4K_120 -> 0
#   HB4K_120 -> 1
#   HN4K_120 -> 2
#   MA4K_120 -> 3
#   MB4K_120 -> 4
#   MN4K_120 -> 5

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
import pickle

MAX_SAMPLES = 200

def save_3d_interactive_figure(embeddings, labels, path_save):
    
    if embeddings.shape[-1] != 3:
        print(f"The shape of the embedding must be 3, now is {embeddings.shape[-1]}.\n")
        exit()
        
    # Define colors for each class
    colors = ['red', 'green', 'blue', 'yellow', "gray", 'black']

    # Create traces for each class
    traces = []
    for c in range(len(np.unique(labels))):  # Iterate over each class
        mask = (labels == c)
        trace = go.Scatter3d(
            x=embeddings[:, 0][mask], 
            y=embeddings[:, 1][mask], 
            z=embeddings[:, 2][mask],
            mode='markers',
            marker=dict(color=colors[c], size=5),
            name=f'Class {c}'
        )
        traces.append(trace)

    # Create layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='t-SNE 1'),
            yaxis=dict(title='t-SNE 2'),
            zaxis=dict(title='t-SNE 3')
        )
    )

    # Plot the figure
    fig = go.Figure(data=traces, layout=layout)

    # Save the interactive plot as an HTML file
    offline.plot(fig, filename=os.path.join(path_save, 'tsne_3d.html'), auto_open=False)


if __name__ == '__main__':
    
    limit=1000
    with open(os.path.join(f"../data/embeddings/limit_{limit}/embeddings.pkl"), 'rb') as f:
        dataset = pickle.load(f)
    # loading the embeddings and its labels
    embeddings = dataset['embeddings']
    demo_labels = dataset['labels']
    eth_labels = dataset['eth_labels']
    gen_labels = dataset['gen_labels']
    
    # extract the reduced ethnicity embeddings from the TSNE
    embeddings_reduced = TSNE(n_components=3).fit_transform(embeddings)
    
    save_3d_interactive_figure(embeddings_reduced, eth_labels, f'task3/eth/limit_{limit}')
    save_3d_interactive_figure(embeddings_reduced, gen_labels, f'task3/gen/limit_{limit}')
    save_3d_interactive_figure(embeddings_reduced, demo_labels, f'task3/demo/limit_{limit}')