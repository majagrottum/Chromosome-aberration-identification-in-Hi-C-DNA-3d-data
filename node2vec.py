import numpy as np

import networkx as nx

from node2vec import Node2Vec


# Defining the node2vec parameters

D = 10
P = 1
Q = 0.5
WL = 300

# Retrieving the adjacency matrices from the .txt files

adjacency_matrices_healthy = np.loadtxt('healthy_hic (1).txt', delimiter=',')

adjacency_matrices_cancer = np.loadtxt('cancer_hic (1).txt', delimiter=',')

# Setting the indices corresponding to the nodes of the different chromosomes

dataset_indices = [(0, 249), (250, 421), (422, 577), (578, 713), (714, 777)]

# Creating the different network graphs corresponding to each chromosome

graphs_healthy = []

graphs_cancer = []

for start, end in dataset_indices:
    
    chromosome_matrix_healthy = adjacency_matrices_healthy[start:end +1, start:end +1]
    
    chromosome_matrix_cancer = adjacency_matrices_cancer[start:end +1, start:end +1]
    
    G_healthy = nx.from_numpy_array(chromosome_matrix_healthy, create_using=nx.DiGraph())
    
    G_cancer = nx.from_numpy_array(chromosome_matrix_cancer, create_using=nx.DiGraph())
    
    graphs_healthy.append(G_healthy)
    
    graphs_cancer.append(G_cancer)


# Creating empty lists for the different embeddings of the two datasets

embeddings_healthy = []

embeddings_cancer = []

for G in graphs_healthy:

    # Precomputing probabilities and generating walks - **ON WINDOWS ONLY WORKS WITH workers=1**

    node2vec_healthy = Node2Vec(G, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=6, p=P, q=Q)  # Use temp_folder for big graphs

    # Generating the node embeddings

    model_healthy = node2vec_healthy.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Retrieving the node embeddings

    embeddings_healthy.append(model_healthy.wv.vectors)
    
for G in graphs_cancer:

    # Precomputing probabilities and generating walks - **ON WINDOWS ONLY WORKS WITH workers=1**

    node2vec_cancer = Node2Vec(G, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=6, p=P, q=Q)  # Use temp_folder for big graphs

    # Generating the node embeddings

    model_cancer = node2vec_cancer.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Retrieving the node embeddings

    embeddings_cancer.append(model_cancer.wv.vectors)

