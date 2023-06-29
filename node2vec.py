import numpy as np

import networkx as nx

from node2vec import Node2Vec


# Defining the node2vec parameters

D = 10
P = 1
Q = 0.5
WL = 300

# Retrieving the adjacency matrices from the .txt files

adjacency_matrices_h = np.loadtxt('healthy_hic (1).txt', delimiter=',')

adjacency_matrices_c = np.loadtxt('cancer_hic (1).txt', delimiter=',')

# Setting the indices corresponding to the nodes of the different chromosomes

dataset_indices = [(0, 249), (250, 421), (422, 577), (578, 713), (714, 777)]

# Creating the different network graphs corresponding to each chromosome

graphs_h = []

graphs_c = []

for start, end in dataset_indices:
    
    chromosome_matrix_h = adjacency_matrices_h[start:end +1, start:end +1]
    
    chromosome_matrix_c = adjacency_matrices_c[start:end +1, start:end +1]
    
    G_h = nx.from_numpy_array(chromosome_matrix_h, create_using=nx.DiGraph())
    
    G_c = nx.from_numpy_array(chromosome_matrix_c, create_using=nx.DiGraph())
    
    graphs_h.append(G_h)
    
    graphs_c.append(G_c)


# Creating empty lists for the different embeddings of the two datasets

embeddings_h = []

embeddings_c = []

for g in graphs_h:

    # Precomputing probabilities and generating walks - **ON WINDOWS ONLY WORKS WITH workers=1**

    node2vec_h = Node2Vec(g, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=6, p=P, q=Q)  # Use temp_folder for big graphs

    # Generating the node embeddings

    model_h = node2vec_h.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Retrieving the node embeddings

    embeddings_h.append(model_h.wv.vectors)
    
for g in graphs_cancer:

    # Precomputing probabilities and generating walks - **ON WINDOWS ONLY WORKS WITH workers=1**

    node2vec_c = Node2Vec(g, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=6, p=P, q=Q)  # Use temp_folder for big graphs

    # Generating the node embeddings

    model_c = node2vec_c.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Retrieving the node embeddings

    embeddings_c.append(model_c.wv.vectors)

