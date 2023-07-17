import numpy as np

import networkx as nx

from node2vec import Node2Vec

# Importing the community module in the python-louvain package 

import community

# Importing HDBSCAN for clustering

import hdbscan

# Importing PCA for dimensionality reduction

from sklearn.decomposition import PCA

# Importing necessary library to create the 2D plots of the transformed (through PCA) embeddings

import matplotlib.pyplot as plt


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

# Defining a function to retrieve the node embeddings using node2vec

def node_embeddings(graph, file_name):

    # Precomputing probabilities and generating walks 
    # Number of workers is set to 1 as it should be smaller than or equal to the number of CPU cores on my laptop which is 2
    
    node2vec = Node2Vec(graph, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=1, p=P, q=Q)  
    
    # Generating the node embedding 
    
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Retrieving the node embedding 
    # model.wv.vectors epresents the embeddings as a two-dimensional array, where each row corresponds to the embedding vector of a specific node.
    # For example, you can retrieve the embedding vector of a specific node using model.wv.vectors[node_index], where node_index is the numeric index of the node (0-based index).
    
    embedding = model.wv.vectors
    
    # The embeddings will be saved in the txt file in a space-separated format, with each row representing the embedding vector of a specific node
    # np.savetxt() will save the file in the current working directory, which is the directory where your Python script or Jupyter Notebook is running
    
    np.savetxt(file_name, embedding, delimiter=" ")

    return embedding


# Retrieving the node embeddings for the healthy cell line

embedding_h = node_embeddings(G_h, "embedding_h.txt")


# Retrieving the node embeddings for the cancer cell line

embedding_c = node_embeddings(G_c, "embedding_c.txt")



    


# Defining a function to retrieve the community labels of each node of the networks

def community_label(graph, file_name):

    # Computing the partition using Louvain community detection
    # The partition dictionary will contain the community assignments for each node in the graph
    # The resolution parameter will change the size of the communities
    # By setting the random_state parameter to the same value each time you run the algorithm, you should obtain the same community assignments
    
    partition = community.best_partition(graph, weight='weight', resolution = 1, random_state = 0))
    
    # Converting the community_assignments dictionary to a numpy array
    # Each tuple from partition_h.items() consists of a node as the key and its corresponding community label as the value.
    # The resulting array will have each tuple as a row, where the first column contains the node and the second column contains the community label.
    
    communities = np.array(list(partition.items()))
    
    # The resulting text file will have each node and its community label on a separate line, separated by a space.
    
    np.savetxt(file_name, communities, delimiter=' ', fmt='%s')

    return partition, communities


# Retrieving the community label of each node for the healthy cell line

partition_h, communities_h = community_label(G_h, 'communities_h.txt')


# Retrieving the community label of each node for the cancer cell line

partition_c, communities_c = community_label(G_c, 'communities_c.txt')





# Clustering is performed below using HDBSCAN



# Defining a function to perform clustering using HDBSCAN

def clustering_HDBSCAN(embedding):
    
    # Generating a clustering object 

    clusterer = hdbscan.HDBSCAN()

    # We can then use this clustering object and fit it to the data we have, which is an array of shape (n_nodes, n_dimensions) representing the node embeddings

    clusterer.fit(embedding)

    # The clusterer object knows, and stores the result in an attribute labels_
    # Access the cluster labels assigned to each data point
    # The result will be an array of cluster labels corresponding to each data point in the embedding
    # Samples that are in the same cluster get assigned the same number. 
    # The cluster labels start at 0 and count up.
    # The shape of the labels array would typically be (num_nodes,), indicating a 1-dimensional array with the length equal to the number of nodes in the dataset.

    labels = clusterer.labels_
    
    # We can thus determine the number of clusters found by finding the largest cluster label and add 1

    num_clusters = clusterer.labels_.max() + 1

    return clusterer, labels, num_clusters



# Importantly HDBSCAN is noise aware – it has a notion of data samples that are not assigned to any cluster. 
# This is handled by assigning these samples the label -1.

# Retrieving the cluster labels assigned to each data point in the node embedding of the healthy cell line

clusterer_h, labels_h, num_clusters_h = clustering_HDBSCAN(embedding_h)

# Retrieving the cluster labels assigned to each data point in the node embedding of the cancer cell line

clusterer_c, labels_c, num_clusters_c = clustering_HDBSCAN(embedding_c)




 
# Dimensionality reduction techniques like PCA can be used to project the node embeddings into a low-dimensional space for better visualization of clusters

# In Principal Component Analysis (PCA) the idea is that data are projected on the first 2 eigenvectors of the covariance matrix, so on the directions which allow most of the variance in the data to be represented.

# Defining a function to perform PCA

def principal_component_analysis(embedding):

    # Creating the PCA Object
    # n_components represents the desired number of principal components to retain
    # we want a bidimensional map
    # n_components=2 means that you are considering the first two eigenvectors (with largest eigenvalues) of the covariance matrix and taking for each point the value of the first eigenvector as X and the value of the second as Y. 

    pca = PCA(n_components=2)

    # Fitting the PCA Model

    pca.fit(embedding)
    
    # Transforming the data
    # The transform method will project the embeddings onto the principal components, resulting in the transformed embeddings.
    
    transformed_embedding = pca.transform(embedding)
    
    return transformed_embedding


# Retrieving the transformed embeddings from the PCA of the node embeddings

# For the healthy cell line

transformed_node_embedding_h = principal_component_analysis(embedding_h)

# For the cancer cell line

transformed_node_embedding_c = principal_component_analysis(embedding_c)






# PCA was applied on the node embedding vectors to visualize them in 2D
# We then use the cluster/chromosome label of each segment to color it 

# Defining a function to create a 2D plot of the transformed embeddings from PCA colored with cluster/chromosome labels

def plot_labels(PCA_embedding, labels, embedding_type, cell_line, label_type):

    # Creating a scatter plot
    # PCA_embedding[:, 0] represents the values of the first principal component, and PCA_embedding[:, 1] represents the values of the second principal component. 
    # The c parameter is set to labels, which assigns a different color to each unique cluster/chromosome label.

    plt.scatter(PCA_embedding[:, 0], PCA_embedding[:, 1], c=labels)

    # Adding labels and title
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Transformed ' + embedding_type + ' Embeddings with ' + label_type + ' Labels for the ' + cell_line)

    # Adding a colorbar
    
    colorbar = plt.colorbar()
    colorbar.set_label(label_type + ' Label')

    # Displaying the plot
    
    plt.show()


# Making the 2D plots of the transformed node embeddings

# For the healthy cell line

plot_labels(transformed_node_embedding_h, labels_h, 'Node', 'Healthy Cell Line', 'Cluster')

# For the cancer cell line

plot_labels(transformed_node_embedding_c, labels_c, 'Node', 'Cancer Cell Line', 'Cluster')





