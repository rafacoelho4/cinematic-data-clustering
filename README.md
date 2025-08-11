# 🎬 In the Mood for Analysis

Uma análise visual e sonora do clássico filme **In the Mood for Love** (2000), de Wong Kar-Wai. O projeto combina técnicas de visão computacional e ciência de dados para agrupar cenas por similaridade, destacando a cinematografia nostálgica e melancólica do diretor. 

<img src="movie_banner.jpg" width="90%" alt="Scene from movie In the Mood for Love (2000)">

---

## 📌 Objetivos

- Extrair automaticamente as cenas do filme.
- Analisar atributos visuais (cor, movimento, brilho) e sonoros (energia, silêncio)
- Detectar magnitude de movimento de câmera ao longo do filme.
- Agrupar cenas com base em suas características utilizando aprendizado não supervisionado (clustering).
- Visualizar padrões recorrentes na estética e ritmo do filme.

---

## 📂 Estrutura do Projeto

- `data/`: Arquivos de entrada (filme, metadados, frames). 
- `notebooks/`: Notebook principal do projeto. 
- `src/`: Funções auxiliares para leitura de vídeo, extração de frames, extração de atributos, etc.
- `figures/`: Gráficos usados no projeto.
- `highlights/`: Vídeos com as cenas agrupadas por clusters.
- `requirements.txt`: Bibliotecas utilizadas.

---

## 🧠 Técnicas Utilizadas

- **OpenCV**: Extração de frames e cálculo de movimento óptico (Farneback).
- **librosa**: Análise de energia e silêncio no áudio.
- **Scikit-learn**: Normalização, clustering com KMeans, PCA para visualização.
- **Matplotlib** / **Plotly**: Visualizações dos atributos e agrupamentos.

---

## 📊 Resultados

- O filme foi dividido em ~X cenas.
- Foram extraídas features como:

   - `hsv_mean`, `hsv_var`: valor médio e variação para Hue, Saturation e Value. 
   - `rgb_mean`, `rgb_var`: valor médio e variação para os canais Red, Green e Blue. 
   
- As cenas foram agrupadas em 4 clusters principais, destacando diferentes atmosferas visuais e sonoras.

<!-- > 🎥 Um vídeo com cenas agrupadas pode ser visto em [`highlights/top_scenes.mp4`](highlights/top_scenes.mp4) -->

---

## 💻 Como executar o projeto

1. Clone o repositório
```bash
   git clone https://github.com/seu-usuario/in-the-mood-analysis.git
   cd in-the-mood-analysis
```

2. Instale os pacotes necessários:
```bash
   pip install -r requirements.txt 
```

3. Execute os notebooks na ordem abaixo:

- 01_scene_detection.ipynb: Divide o filme em cenas
- 02_feature_extraction.ipynb: Extrai atributos de vídeo e som
- 03_clustering.ipynb: Aplica PCA/UMAP e KMeans
- 04_visualization.ipynb: Gera gráficos e salva cenas agrupadas

--- 

## 🔮 Possíveis Extensões

- Aplicar o pipeline em outros filmes da filmografia de Wong Kar-Wai. 
- Classificação automática de cenas por tom emocional. 

--- 

## 👨‍💻 Autor

Rafael Coelho Monte Alto 
Bacharel em Ciência da Computação — UFOP 
Interessado em Pesquisa Operacional, Ciência de Dados e Visão Computacional. 

[https://www.linkedin.com/in/rafael-coelho-alto/][LinkedIn] 
