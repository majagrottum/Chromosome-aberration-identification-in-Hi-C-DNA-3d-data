import numpy as np

# Importing library for creating network graphs

import networkx as nx

# Importing library for using the node2vec algorithm for node embedding

from node2vec import Node2Vec

# Importing library for using the HDBSCAN algorithm for clustering

import hdbscan

# Importing PCA for dimensionality reduction

from sklearn.decomposition import PCA

# Importing necessary library to create the 2D plots of the transformed (through PCA) embeddings

import matplotlib.pyplot as plt


# Defining a function to retrieve an adjacency matrix from a .txt file
# The filename is the name of the file in string format
# splitting defines how the values in the file are separated, e.g. comma separated values: ','

def adjacency_matrix_from_file(filename, splitting):

    adjacency_matrix = np.loadtxt(filename, delimiter = splitting)

    return adjacency_matrix


# Defining a function to create a network graph from an adjacency matrix

def create_graph(adjacency_matrix):

    # nx.from_numpy_matrix(A, create_using=None)returns a graph from a numpy matrix.
    # The numpy matrix is interpreted as an adjacency matrix for the graph.
    
    graph = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.Graph())

    return graph



# Defining a function to remove isolated nodes from network graphs 

def remove_isolated_nodes(graph):

    isolated_nodes_list = list(nx.isolates(graph))

    graph_new = graph.remove_nodes_from(isolated_nodes_list)

    return graph_new






# Below follows the node2vec algorithm

# Defining a function to retrieve the node embeddings using node2vec

def node_embeddings(graph, file_name, D, P, Q, WL):

    # Precomputing probabilities and generating walks 
    # Number of workers is set to 1 as it should be smaller than or equal to the number of CPU cores on your computer
    
    node2vec = Node2Vec(graph, dimensions=D, walk_length=WL, num_walks=10, weight_key='weight', workers=1, p=P, q=Q)  
    
    # Generating the node embedding 
    
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Saving embeddings for later use
    # Each row in the txt file corresponds to one node, the first column corresponds to the index of the node within the network, and then follows the 10 coordinates of that node
      
    model.wv.save_word2vec_format(file_name)

    

# Defining a function to retrieve node embeddings from a txt file

def embedding_dictionary(file_name):
    
    with open(file_name, "r") as file:
        
        # Skipping the first line, as this is just telling us that the embedding is composed of x number of nodes, each one characterized by y coordinates
        
        file.readline()
        
        # Read the contents of the file

        content = file.readlines()
        
    data = {}
    
    for line in content:

        # Separating each line in the file based on the separator (here being space)
        
        elements = line.split()

        # Finding the node 

        label = int(elements[0])

        # Making a list for the embeddings
        
        coordinates = []
        
        for c in elements[1:]:
            
                coordinates.append(float(c))

        # The dictionary data then contains the nodes as keys and their embeddings (in a list) as values

        data[label] = coordinates
        
    # Using the sorted() function to sort the items in the dictionary based on the keys. 
    # The lambda function lambda x: x[0] specifies that the sorting should be done based on the first element (x[0]) of each key-value pair.
    # The sorted() function returns a list of sorted key-value pairs, which we then convert back into a dictionary using the dict() function. 
    # The resulting sorted_data dictionary will have the keys sorted in increasing order.
    
    sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
    
    return sorted_data



    


# Clustering is performed below using HDBSCAN


# Defining a function to perform clustering using HDBSCAN

def clustering_HDBSCAN(embedding):
    
    # Generating a clustering object 

    clusterer = hdbscan.HDBSCAN()

    # We can then use this clustering object and fit it to the data we have, which is an array of shape (n_nodes, n_dimensions) representing the node embeddings

    clusterer.fit(embedding)

    # labels will be an array of cluster labels corresponding to each data point in the embedding
    # Samples that are in the same cluster get assigned the same number. 
    # Importantly HDBSCAN is noise aware – it has a notion of data samples that are not assigned to any cluster. 
    # This is handled by assigning these samples the label -1.
    # The shape of the labels array would typically be (num_nodes,), indicating a 1-dimensional array with the length equal to the number of nodes in the dataset.

    labels = clusterer.labels_

    return labels





 
# Dimensionality reduction techniques like PCA can be used to project the node embeddings into a low-dimensional space for better visualization of clusters

# In Principal Component Analysis (PCA) the idea is that data are projected on the first 2 eigenvectors of the covariance matrix, so on the directions which allow most of the variance in the data to be represented.

# Defining a function to perform PCA

def principal_component_analysis(embedding):

    # Creating the PCA Object
    # n_components represents the desired number of principal components to retain
    # n_components=2 means that you are considering the first two eigenvectors (with largest eigenvalues) of the covariance matrix and taking for each point the value of the first eigenvector as X and the value of the second as Y. 

    pca = PCA(n_components=2)

    # Fitting the PCA Model

    pca.fit(embedding)
    
    # Transforming the data by projecting the embeddings onto the principal components
    
    transformed_embedding = pca.transform(embedding)
    
    return transformed_embedding






# PCA was applied on the node embedding vectors to visualize them in 2D
# We can then use the cluster label of each segment to color it 

# Defining a function to create a 2D plot of the transformed embeddings from PCA colored with cluster labels

def plot_cluster_labels(PCA_embedding, labels):
    
    # Creating a scatter plot
    # PCA_embedding[:, 0] represents the values of the first principal component, and PCA_embedding[:, 1] represents the values of the second principal component. 
    # The c parameter is set to labels, which assigns a different color to each unique cluster label.

    plt.scatter(PCA_embedding[:, 0], PCA_embedding[:, 1], c=labels, cmap='viridis')

    # Adding labels and title
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Transformed Node Embeddings with Cluster Labels')

    # Adding a colorbar
    
    colorbar = plt.colorbar()
    colorbar.set_label('Cluster Label')

    # Displaying the plot
    
    plt.show()
