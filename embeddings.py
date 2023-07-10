import numpy as np

import networkx as nx

from node2vec import Node2Vec

# Importing the community module in the python-louvain package (for the community embedding)

import community


# Retrieving the adjacency matrices from the .txt files

adjacency_matrix_h = np.loadtxt('healthy_hic (1).txt', delimiter=',')

adjacency_matrix_c = np.loadtxt('cancer_hic (1).txt', delimiter=',')


# Creating the different network graphs corresponding to each cell line (healthy/cancer)
# from_numpy_matrix(A, create_using=None)returns a graph from numpy matrix.
# The numpy matrix is interpreted as an adjacency matrix for the graph.

G_h = nx.from_numpy_matrix(adjacency_matrix_h, create_using=nx.Graph())
    
G_c = nx.from_numpy_matrix(adjacency_matrix_c, create_using=nx.Graph())


# Before applying node2vec i remove the isolated nodes from the networks (rows/columns with zero sum)

# list(nx.isolates(G)) is a list of all isolated nodes

G_h.remove_nodes_from(list(nx.isolates(G_h)))

G_c.remove_nodes_from(list(nx.isolates(G_c)))



# Below follows the node2vec algorithm, which should be run separately (at separate times) for the two cell lines due to the analysis being heavy in terms of computational cost


# Defining the node2vec parameters

D = 10
P = 1
Q = 0.5
WL = 300


# Precomputing probabilities and generating walks for the healthy cell line
# Number of workers is set to 1 as it should be smaller than or equal to the number of CPU cores on my laptop which is 2

node2vec_h = Node2Vec(G_h, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=1, p=P, q=Q)  

# Generating the node embedding for the healthy cell line

model_h = node2vec_h.fit(window=10, min_count=1, batch_words=4)

# Retrieving the node embedding for the healthy cell line
# model.wv.vectors epresents the embeddings as a two-dimensional array, where each row corresponds to the embedding vector of a specific node.
# For example, you can retrieve the embedding vector of a specific node using model.wv.vectors[node_index], where node_index is the numeric index of the node (0-based index).

embedding_h = model_h.wv.vectors

# The embeddings will be saved in the txt file in a space-separated format, with each row representing the embedding vector of a specific node
# np.savetxt() will save the file in the current working directory, which is the directory where your Python script or Jupyter Notebook is running

np.savetxt("embedding_h.txt", embedding_h, delimiter=" ")


# Precomputing probabilities and generating walks for the cancer cell line
# Number of workers is set to 1 as it should be smaller than or equal to the number of CPU cores on my laptop which is 2

node2vec_c = Node2Vec(G_c, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=1, p=P, q=Q) 

# Generating the node embedding for the cancer cell line

model_c = node2vec_c.fit(window=10, min_count=1, batch_words=4) 

# Retrieving the node embedding for the cancer cell line
# model.wv.vectors epresents the embeddings as a two-dimensional array, where each row corresponds to the embedding vector of a specific node.
# For example, you can retrieve the embedding vector of a specific node using model.wv.vectors[node_index], where node_index is the numeric index of the node (0-based index).

embedding_c = model_c.wv.vectors

# The embeddings will be saved in the txt file in a space-separated format, with each row representing the embedding vector of a specific node
# np.savetxt() will save the file in the current working directory, which is the directory where your Python script or Jupyter Notebook is running

np.savetxt("embedding_c.txt", embedding_c, delimiter=" ")



    

# Below follows the community embedding of the networks of the two cell lines using Louvain Community Detection
# First for the healthy cell line, then for the cancer cell line

# Retrieving the community label of each node for the healthy cell line

# Computing the partition using Louvain community detection
# The partition dictionary will contain the community assignments for each node in the graph

partition_h = community.best_partition(G_h, weight='weight')

# Converting the community_assignments dictionary to a numpy array
# Each tuple from partition_h.items() consists of a node as the key and its corresponding community label as the value.
# The resulting array will have each tuple as a row, where the first column contains the node and the second column contains the community label.

communities_h = np.array(list(partition_h.items()))

# The resulting text file will have each node and its community label on a separate line, separated by a space.

np.savetxt('communities_h.txt', communities_h, delimiter=' ', fmt='%s')


# Retrieving the community label of each node for the cancer cell line

# Computing the partition using Louvain community detection
# The partition dictionary will contain the community assignments for each node in the graph

partition_c = community.best_partition(G_c, weight='weight')

# Converting the community_assignments dictionary to a numpy array
# Each tuple from partition_h.items() consists of a node as the key and its corresponding community label as the value.
# The resulting array will have each tuple as a row, where the first column contains the node and the second column contains the community label.

communities_c = np.array(list(partition_c.items()))

# The resulting text file will have each node and its community label on a separate line, separated by a space.

np.savetxt('communities_c.txt', communities_c, delimiter=' ', fmt='%s')
