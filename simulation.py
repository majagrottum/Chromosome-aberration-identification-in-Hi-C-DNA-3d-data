import numpy as np

import networkx as nx

from node2vec import Node2Vec

# Importing HDBSCAN for clustering

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



# Defining a function that maps the old indices of nodes into the new ones after having removed the isolated nodes from a network graph

def create_node_mapping(graph):

    remaining_nodes = list(graph.nodes)
    
    mapping = {}
    
    for index, node in enumerate(remaining_nodes):
        
        mapping[node] = index

    # The function returns a dictionary where the key is old index and the value is the new index
    
    return mapping



# Defining functions to retrieve the new index of a node after removal of isolated nodes from a network graph

def get_new_node_index_start(mapping, old_index):
    
    try:
        
        return mapping[old_index]
    
    except KeyError:
        
        # If the key is not found in the mapping dictionary, find the closest higher key
        # Finds the smallest key in the mapping dictionary that is greater than the given key
        
        closest_higher_key = min(filter(lambda x: x > old_index, mapping.keys()))
        
        return mapping[closest_higher_key]
        
    
def get_new_node_index_end(mapping, old_index):
    
    try:
        
        return mapping[old_index]
    
    except KeyError:
        
        # If the key is not found in the mapping dictionary, find the closest smaller key
        # Finds the highest key in the mapping dictionary that is smaller than the given key
        
        closest_smaller_key = max(filter(lambda x: x < old_index, mapping.keys()))
        
        return mapping[closest_smaller_key]



# Defining a function to retrieve the new indices corresponding to the start and end nodes of the different sets of nodes in the network
# Sets of nodes correspond to different chromosomes in the case of Hi-C data

def new_indices_of_sets(mapping, graph, old_indices):

    # Creating a list with the new indices corresponding to the different sets of nodes

    new_dataset_indices = []
    
    for start, end in old_indices:
        
        new_start = get_new_node_index_start(mapping, start)
        
        new_end = get_new_node_index_end(mapping, end)
        
        new_dataset_indices.append((new_start, new_end))

    return new_dataset_indices



# Defining a function that returns a list of the corresponding label of each node according to the different sets of nodes (in this case chromosomes)

# dataset_indices is on the same form as the list returned from new_indices_of_sets

# The label_mapping dictionary is defined to map the original label indices to the desired customized labels. 
# An example of the argument is label_mapping = {0: 1, 1: 6, 2: 'X', 3: 10, 4: 20}
# In this example the label_mapping dictionary maps index 0 to label 1, index 1 to label 6, index 2 to label 'X', index 3 to label 10, and index 4 to label 20.
# This dictionary has to contain the same number of elements as the dataset_indices

def nodes_labeled_as_chromosomes(dataset_indices, label_mapping):

    chromosome_labels = []
    
    for i, (start, end) in enumerate(dataset_indices):

        # The first argument passed to get() is the key i that is being looked up in the dictionary.
        # The second argument passed to get() is a default value that will be returned if the key i is not found in the dictionary.
       
        label = label_mapping.get(i, i)

        # Adds the label x times where x is the number of nodes corresponding to that label

        chromosome_labels += ([label] * (end - start + 1))
        
    return chromosome_labels






# Below follows the node2vec algorithm

# Defining the node2vec parameters used for the example of the Hi-C data

D = 10
P = 1
Q = 0.5
WL = 300

# Defining a function to retrieve the node embeddings using node2vec

def node_embeddings(graph, file_name):

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





