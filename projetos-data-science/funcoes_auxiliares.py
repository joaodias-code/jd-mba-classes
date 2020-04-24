import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def demonstracao_pca(df, n_components):
    # Aplica o PCA aos dados usando uma quantidade passada de componentes
    pca = PCA(n_components=n_components).fit(df)
    
    # PCA Dimensões
    dimensoes = ['Dimensão {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA Componentes
    componentes = pd.DataFrame(np.round(pca.components_, 4), columns=df.keys())
    componentes.index = dimensoes
    
    # PCA Variância explicada
    taxa = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    taxa_variancia = pd.DataFrame(np.round(taxa, 4), columns = ['Variância Explicada'])
    taxa_variancia.index = dimensoes

    # Cria a visualização 
    fig, ax = plt.subplots(figsize=(14,8))
    componentes.plot(ax=ax, kind = 'bar');
    ax.set_ylabel("Peso das variáveis")
    ax.set_xticklabels(dimensoes, rotation=0)

    # Display the explained variance ratios
    for i, variancia_explicada in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.70, ax.get_ylim()[1] + 0.05, "Variância Explicada\n %.4f"%(variancia_explicada))

        
        
def DBSCAN_Clusters(df, lista_eps, min_samples):
    # Gera as compinações de hiperparâmetros
    combinacoes = [(eps, min_samples) for eps in lista_eps]
    
    # Estabelece um limite de 4 combinações devido a limitação de 4 subplots no grid
    if len(combinacoes) > 4:
        print('Passe no máximo 4 valores de epsilon')
        return
    
    # Função pra construir o modelo
    construir_modelo = lambda eps, min_samples: DBSCAN(eps=eps, metric='euclidean', min_samples=min_samples, n_jobs=-1)
    
    # Armazena as clusterizações
    clusters = [construir_modelo(combinacao[0], combinacao[1]).fit_predict(df) for combinacao in combinacoes]
    
    # Separa as dimensões para os gráficos
    x = df['Dimensão 1']
    y = df['Dimensão 2']
    
    # Define o grid de visualizações
    fig, ((ax1,ax2) ,(ax3,ax4)) = plt.subplots(2, 2, figsize=(12,8))
   
    # Constrói os subplots
    for ax, combinacao, cluster in zip(fig.get_axes(), combinacoes, clusters):
        ax.scatter(x, y, c=cluster, cmap=cm.get_cmap('rainbow'))
        ax.set_title('Eps = %.1f, Min_Pts = %d' % (combinacao[0], combinacao[1]))
        ax.set(xlabel='Dimensão 1', ylabel='Dimensão 2')
        ax.label_outer()
