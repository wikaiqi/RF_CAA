import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(data, label, game_loss, game):

    if not os.path.exists(Path('pictures')):
        os.makedirs(Path('pictures'))
    print(data.shape)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    df = pd.DataFrame({'x': tsne_results[:,0], 'y': tsne_results[:, 1]})
    df['label'] = label
    df_1 = df[df['label']==1].copy()
    df_0 = df[df['label']==0].copy()

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(df_0['x'].values, df_0['y'].values, c='orange', s=50, label='label = 0')
    plt.scatter(df_1['x'].values, df_1['y'].values, c='cyan', s=50, label='label = 1')
    plt.legend()
    plt.grid(True)
    if game:
        plt.title("TSNE plot: Three-way Game")
        plt.savefig('pictures/tsne_CAA.png')
    else:
        plt.title("TSNE plot: CNN")
        plt.savefig('pictures/tsne_CNN.png')

    plt.close()
    if game:
        epoch = range(1, len(game_loss)+1)
        plt.figure(figsize=(12, 8))
        plt.plot(epoch, game_loss)
        plt.xlabel('epoch')
        plt.ylabel('adversarial loss')
        plt.savefig('pictures/CAA_adversarial_loss.png')
    
