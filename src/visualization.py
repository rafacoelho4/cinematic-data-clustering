import pandas as pd 
import matplotlib as plt 
import seaborn as sns

# Image related 
def plot_hue_saturation(color_feats):
    # Gráfico 1: distribuição dos valores de Hue médio nos shots
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    sns.histplot(color_feats['hue_mean'], bins=30, kde=True, color='darkred')
    plt.title("Distribuição de Hue médio por shot")
    plt.xlabel("Hue médio (0 a 180)")
    plt.ylabel("Número de shots")

    # Gráfico 2: distribuição da Saturação média
    plt.subplot(1,2,2)
    sns.histplot(color_feats['sat_mean'], bins=30, kde=True, color='goldenrod')
    plt.title("Distribuição de Saturação média por shot")
    plt.xlabel("Saturação média (0 a 255)")
    plt.ylabel("Número de shots")
    plt.tight_layout()
    plt.show()

    # Gráfico 3: scatter Hue vs Saturação
    plt.figure(figsize=(6,6))
    plt.scatter(color_feats['hue_mean'], color_feats['sat_mean'], c=color_feats['shot_id'], cmap='winter', s=60)
    plt.colorbar(label='shot_id')
    plt.xlabel('Hue médio')
    plt.ylabel('Saturação média')
    plt.title('Hue × Saturação por shot')
    plt.grid(False)
    plt.show()

# Audio related 
def plot_rms_silence(df_shots):
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))

    # 1. RMS médio vs shot_id
    ax = axes[0]
    sns.lineplot(x='shot_id', y='rms_mean', data=df_shots, marker='o', ax=ax, color='royalblue')
    ax.set_title('RMS médio por cena'); ax.set_xlabel('Shot ID'); ax.set_ylabel('RMS médio')

    # 2. duplo eixo: RMS médio e Silence ratio
    ax = axes[1]
    sns.lineplot(x='shot_id', y='rms_mean', data=df_shots, marker='o', ax=ax, color='royalblue', label='RMS médio')
    ax2 = ax.twinx()
    sns.lineplot(x='shot_id', y='silence_ratio', data=df_shots, marker='x', ax=ax2, color='orange', label='Silêncio (%)')
    ax.set_title('Volume vs Silêncio'); ax.legend(loc='upper left'); ax2.legend(loc='upper right')
    ax.set_xlabel('Shot ID'); ax.set_ylabel('RMS médio'); ax2.set_ylabel('Silêncio')

    plt.tight_layout()
    plt.show()

# Movement related 
def plot_movement(df_shots):
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))

    sns.lineplot(x='shot_id', y='flow_mag_mean', data=df_shots, marker='o', ax=ax, color='royalblue')
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