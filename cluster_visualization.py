import numpy as np
import networkx as nx
from node2vec import Node2Vec
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def create_graph(filename, splitting):

    """This function creates a network graph from an adjacency matrix stored in a .txt file.
    The content in the .txt file is assumed to make a symmetric matrix containing numerical 
    values representing the presence or absence of edges and their corresponding weights.
       
    Parameters:
    
        filename: name of the file the adjacency matrix is stored in (string format)
        splitting: defines how the values in the file are separated (string format), e.g. for comma separated values this parameter is ','

    Returns:
    
         A NetworkX graph object created from the adjacency matrix (numpy matrix)."""

    adjacency_matrix = np.loadtxt(filename, delimiter = splitting)
    
    graph = nx.from_numpy_matrix(adjacency_matrix, create_using=nx.Graph())

    return graph




def remove_isolated_nodes(graph):

    """ This function removes the isolated nodes from a network graph

    Parameters:

        graph: a NetworkX graph object

    Returns:

        A new NetworkX graph object where the isolated nodes in the input parameter graph have been removed."""

    isolated_nodes_list = list(nx.isolates(graph))

    graph_new = graph.remove_nodes_from(isolated_nodes_list)

    return graph_new




def node_embeddings(graph, file_name, D, WL, NW, P, Q):

    """ This function compute node embeddings using the node2vec algorithm (graph is assumed to be weighted) and stores the result in a .txt file. 
        Each row in the .txt file will correspond to one node, the first column corresponds to the index of the node within the network, 
        and then follows the D coordinates of that node.

    Parameters:

        graph: a NetworkX graph object
        file_name: name of the file the node embeddings will be stored in (string format)
        D: embedding dimensions
        WL: number of nodes in each walk
        NW: number of walks per node
        P: return hyper parameter
        Q: inout parameter
        
    """

    # Precomputing probabilities and generating walks 
    # Number of workers is set to 1 as it should be smaller than or equal to the number of CPU cores on your computer
    node2vec = Node2Vec(graph, dimensions=D, walk_length=WL, num_walks=NW, weight_key='weight', workers=1, p=P, q=Q)  
    
    # Generating the node embedding 
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Saving embeddings for later use
    model.wv.save_word2vec_format(file_name)




def embedding_dictionary(file_name):

    """This function retrieves node embeddings stored in a .txt file.
    
        The content in the .txt file needs to be in a space separated form.
        The first row in the file should tell that the embedding is composed of x number of nodes, each one characterized by y coordinates,
        with x being in the first coloumn and y being in the second coloumn.
        For the next rows should each row correspond to one node, the first column corresponding to the index of the node within the network, 
        and then follows the coordinates of that node (number of coordinates corresponds to the embedding dimension).
        
    Parameters:

        file_name: name of the file the node embeddings are stored in (string format)

    Returns:

        A sorted dictionary (keys in increasing order) where the keys are the nodes and the values are their corresponding embeddings.
        Each value in the dictionary is thus the embedding corresponding to a node stored in a list. """
    
    with open(file_name, "r") as file:
        
        # Skipping the first line, as this is just telling us the number of nodes and embedding dimension
        file.readline()
        
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
        
    # The lambda function lambda x: x[0] specifies that the sorting should be done based on the first element (x[0]) of each key-value pair.
    # The sorted() function returns a list of sorted key-value pairs, which we then convert back into a dictionary using the dict() function. 
    sorted_data = dict(sorted(data.items(), key=lambda x: x[0]))
    
    return sorted_data

    


def clustering_HDBSCAN(embedding):

    """This function performs clustering on node embeddings using the HDBSCAN algorithm and finds the cluster label associated to each node

    Parameters:

        embedding: the node embeddings given as a list, where each element in the list is a list with the embedding corresponding to one node

    Returns:

        An ndarray of shape (n_samples,) containing the cluster labels for each point in the dataset.
        
        Samples that are in the same cluster get assigned the same number. 
        HDBSCAN is noise aware – it has a notion of data samples that are not assigned to any cluster. 
        This is handled by assigning these samples the label -1."""
    
    # Generating a clustering object 
    clusterer = hdbscan.HDBSCAN()

    # Using the clustering object and fitting it to the data we have
    clusterer.fit(embedding)

    # labels will be an array of cluster labels corresponding to each data point in the embedding
    labels = clusterer.labels_

    return labels




def principal_component_analysis(embedding):

    """This function performs principal component analysis to project node embeddings into a two-dimensional space.

    Parameters:

        embedding: the node embeddings given as a list, where each element in the list is a list with the embedding corresponding to one node

    Returns: 

        A ndarray of shape (n_samples, n_components) containing the transformed embeddings after PCA, 
        where n_samples is the number of nodes and n_components = 2."""

    # Creating the PCA Object
    # n_components represents the desired number of principal components to retain
    pca = PCA(n_components=2)

    # Fitting the PCA Model
    pca.fit(embedding)
    
    # Transforming the data by projecting the embeddings onto the principal components
    transformed_embedding = pca.transform(embedding)
    
    return transformed_embedding




def plot_cluster_labels(PCA_embedding, labels):

    """This function creates a 2D plot of transformed node embeddings from PCA where different colors in the plot correspond to different cluster labels.
    The plot then shows the different clusters present in a network.

    Parameters:

        PCA_embedding: transformed node embeddings from PCA, which is a ndarray of shape (n_samples, n_components), 
        where n_samples is the number of nodes and n_components = 2
        
        labels: a ndarray of shape (n_samples,) containing the cluster labels for each node in the dataset.
        
    """
    
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
