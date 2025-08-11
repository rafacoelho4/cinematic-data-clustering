import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_color_features(color_feats):
    plt.figure(figsize=(18, 10))
    
    # --------------------------------------------
    # Row 1: HSV Distributions (Hue, Saturation, Value)
    # --------------------------------------------
    plt.subplot(2, 3, 1)
    sns.histplot(color_feats['hue_mean'], bins=30, kde=True, color='darkred')
    plt.title("Hue Mean Distribution")
    plt.xlabel("Hue (0-180)")
    # X-axis: Hue values (0° to 180°) in OpenCV’s HSV scale, where:
    # 0° = Red, 30° = Yellow, 60° = Green, 90° = Cyan, 120° = Blue, 150° = Magenta, 180° = Red again.
    
    plt.subplot(2, 3, 2)
    sns.histplot(color_feats['sat_mean'], bins=30, kde=True, color='goldenrod')
    plt.title("Saturation Mean Distribution")
    plt.xlabel("Saturation (0-255)")
    
    plt.subplot(2, 3, 3)
    sns.histplot(color_feats['val_mean'], bins=30, kde=True, color='black')
    plt.title("Brightness Mean Distribution")
    plt.xlabel("Brightness (0-255)")
    
    # --------------------------------------------
    # Row 2: RGB Distributions and Key Relationships
    # --------------------------------------------
    plt.subplot(2, 3, 4)
    sns.kdeplot(data=color_feats[['r_mean', 'g_mean', 'b_mean']], 
                palette=['red', 'green', 'blue'], 
                fill=True, alpha=0.3)
    plt.title("RGB Channel Intensity Distribution")
    plt.xlabel("Intensity (0-255)")

    # RGB means boxplot 
    plt.subplot(2, 3, 5)
    sns.boxplot(data=color_feats[['r_mean', 'g_mean', 'b_mean']],
                palette=['red', 'green', 'blue'])
    plt.title("RGB Channel Means")
    plt.ylabel("Intensity (0-255)")

    # Leave the 6th subplot empty or remove it
    plt.subplot(2, 3, 6)
    plt.axis('off') # Hide empty subplot
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/hsv_rgb_values.jpeg")

    plt.tight_layout()
    plt.show()

# Audio related 
def plot_rms_silence(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    # 1. RMS médio vs shot_id
    ax = axes[0]
    sns.lineplot(x='shot_id', y='rms_mean', data=df, marker='o', ax=ax, color='royalblue')
    ax.set_title('RMS médio por cena'); ax.set_xlabel('Shot ID'); ax.set_ylabel('RMS médio')

    # 2. duplo eixo: RMS médio e Silence ratio
    ax = axes[1]
    sns.lineplot(x='shot_id', y='rms_mean', data=df, marker='o', ax=ax, color='royalblue', label='RMS médio')
    ax2 = ax.twinx()
    sns.lineplot(x='shot_id', y='silence_ratio', data=df, marker='x', ax=ax2, color='orange', label='Silêncio (%)')
    ax.set_title('Volume vs Silêncio'); ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax.set_xlabel('Shot ID'); ax.set_ylabel('RMS médio'); ax2.set_ylabel('Silêncio')

    plt.tight_layout()
    plt.show()

# Movement related 
def plot_movement(df):
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))

    sns.lineplot(x='shot_id', y='flow_mag_mean', data=df, marker='o', ax=ax, color='royalblue')
    ax.set_title('Movimento da cena'); ax.set_xlabel('Cena ID'); ax.set_ylabel('Flow Mag Mean')
    plt.show()

# Cluster related 
def plot_scatter_pca(df):
    # Visualização do scatter PCA colorido por cluster
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='tab10', s=60)
    plt.title('PCA 2D + KMeans Clusters')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(title='Cluster')
    plt.show()

def plot_scatter_umap(df):
    # Visualização do scatter UMAP colorido por cluster
    plt.figure(figsize=(6,5))
    sns.scatterplot(x='umap1', y='umap2', hue='cluster', data=df, palette='tab10', s=60)
    plt.title('UMAP 2D + KMeans Clusters')
    plt.xlabel('UMAP1'); plt.ylabel('UMAP2'); plt.legend(title='Cluster')
    plt.show()

def plot_cluster_timeline(df, n_clusters=None):
    """
    Plota uma timeline horizontal de cada cena colorida por cluster.
    df deve ter colunas ['start','end','cluster'].
    """
    if n_clusters is None:
        n_clusters = df['cluster'].nunique()

    plt.figure(figsize=(10, 1 + n_clusters*0.5))
    for cid in sorted(df['cluster'].unique()):
        subset = df[df['cluster']==cid]
        for _, row in subset.iterrows():
            plt.hlines(y=cid,
                       xmin=row.start,
                       xmax=row.end,
                       color=f"C{cid}",
                       linewidth=6)
    plt.xlabel("Tempo (s)")
    plt.yticks(sorted(df['cluster'].unique()), [f"Cluster {i}" for i in sorted(df['cluster'].unique())])
    plt.title("Timeline de cenas por cluster")
    plt.tight_layout()
    plt.show()