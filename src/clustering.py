from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

import umap.umap_ as umap

import matplotlib.pyplot as plt 

def silhouette(X_scaled):
    """
    Testa diferentes quantidades de clusters e avalia cada um com silhouette_score. 

    Arugumentos:
        X_scaled (DataFrame): dados a serem agrupados. 

    Imprime:
        Valor de silhouette_score para cada quantidade de clusters e àquela quantidade que gerou o melhor resultado (mais perto de 1.0).
    """
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

def clustering(df, video_name="data/snippet.mp4", method="kmeans", n_clusters=4):

    # 1. Escolher apenas as colunas de features para clustering 
    feature_cols = [
        'hue_mean', 'sat_mean', 'val_mean', 
        'hue_var', 'sat_var', 'val_var', 
        'r_mean', 'g_mean', 'b_mean', 
        'r_var', 'g_var', 'b_var' 

        # 'rms_mean', 'rms_std', 'silence_ratio', 
        # 'flow_mag_mean'
        ]
    X = df[feature_cols].values

    # 2. Normalizar com StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Redução de dimensionalidade
    # 3a. PCA para 2D
    pca = PCA(n_components=2, random_state=42)
    emb_pca = pca.fit_transform(X_scaled)
    df['pca1'], df['pca2'] = emb_pca[:,0], emb_pca[:,1]

    # 3b. UMAP para 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_umap = reducer.fit_transform(X_scaled)
    df['umap1'], df['umap2'] = emb_umap[:,0], emb_umap[:,1]

    if(method == "kmeans"): 
        return kmeans_clustering(df, X_scaled, n_clusters) 
    elif(method == "dbscan"): 
        return dbscan_clustering(df, X_scaled) 
    else: 
        return kmeans_clustering(df, X_scaled, n_clusters) 

    # Plotting the clusters
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # K-means plot
    # ax1.scatter(df['pca1'], df['pca2'], c=kmeans_labels, cmap='viridis')
    # ax1.set_title('K-means Clustering')

    # DBSCAN plot
    # ax2.scatter(df['pca1'], df['pca2'], c=dbscan_labels, cmap='viridis')
    # ax2.set_title('DBSCAN Clustering')

    # plt.show()

    return df 

def kmeans_clustering(df, X_scaled, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    return df 

def dbscan_clustering(df, X_scaled): 
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    df['cluster'] = dbscan.fit_predict(X_scaled)
    df['cluster'] = df['cluster'] + 1

    return df 