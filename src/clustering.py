import pandas as pd 
import matplotlib as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import umap.umap_ as umap

def silhouette(X_scaled):
    inertias = []
    silhouettes = []

    start_range = 4
    end_range = 10
    K_range = range(start_range, end_range)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    
    for idx, val in enumerate(silhouettes): 
        print(idx + start_range, val) 
    # print(silhouettes) 
    print("Best n_clusters by silhouette score:", silhouettes.index(max(silhouettes)) + start_range)

def clustering(video_name="data/snippet.mp4", n_clusters=4):
    # 2. Carregar DataFrame com features já extraídas:
    #    Suponha que df_shots contenha columns:
    #    ['shot_id','start','end','duration',
    #     'flow_mag_mean','rms_mean','silence_ratio','hue_mean','sat_mean', ...]

    df = pd.read_csv(f'data/{video_name}.csv')

    # 3. Escolher apenas as colunas de features para clustering
    feature_cols = [
        'hue_mean', 'hue_var', 'sat_mean', 'sat_var', 'val_mean',
        'val_var', 'r_mean', 'r_var', 'g_mean', 'g_var', 'b_mean', 'b_var'

        # 'rms_mean', 'rms_std', 'silence_ratio', 
        # 'hue_mean', 'hue_var',
        # 'sat_mean', 'sat_var', 
        # 'flow_mag_mean'
        ]
    X = df[feature_cols].values

    # 4. Normalizar com StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Redução de dimensionalidade
    # 5a. PCA para 2D
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = emb_pca[:,0], emb_pca[:,1]

    # 5b. UMAP para 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_umap = reducer.fit_transform(X_scaled)
    df['umap1'], df['umap2'] = emb_umap[:,0], emb_umap[:,1]

    # 6. Clustering (K-Means)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # silhouette(X_scaled)
    return df 