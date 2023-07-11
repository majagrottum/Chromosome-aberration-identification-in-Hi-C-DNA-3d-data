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



    
# Below follows the community embedding of the networks of the two cell lines using Louvain Community Detection


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


# Community embedding is the vector that has in each element the number/fraction of first neighbors being in a community (first element for neighbours in community 1, second element for com 2, etc).
# Suppose node X has 10 neighbours: 5 in community A, 3 in community B, 0 in community C, 2 in community D. 
# Then the embedding for node X is [0.5, 0.3, 0, 0.2, 0, ..., 0] (long as the number of communities in the network)

# In this fully connected network, each node has all the other nodes as first neighbours
# We should use a threshold in weights to select who are the real neighbors

threshold_weights = 10000


# Defining a function for the community embedding of the networks

def community_embedding(partition, threshold, communities, graph, file_name):
    
    # Retrieving the number of communities

    unique_communities = (np.unique(list(partition.values()))).tolist()

    num_communities = len(unique_communities)


    # Creating an empty array for the community embeddings

    community_embeddings = []
    
    for node, community in communities:

        neighbors = []

        for neighbor in graph.neighbors(node):

            # Find neighbors above the weight threshold

            if graph[node][neighbor]['weight'] > threshold:

                neighbors.append(neighbor)

        # Creating array for counts of the different communities

        community_counts = [0] * num_communities

        for n in neighbors:

            # Getting the community of a specific neighbor node

            community_neighbor = partition[n]

            community_index = unique_communities.index(community_neighbor)

            community_counts[community_index] += 1

        # Calculating the number of neighbors

        num_neighbors = len(neighbors)
        
        if num_neighbors > 0:
            
            # Creating an empty array for the community embedding of the node

            embedding_node = [] * num_communities

            for count in community_counts:

                embedding_node.append(count/len(neighbors))
                
        else:
            
            # Creating an empty array for the community embedding of the node

            embedding_node = [0.0] * num_communities
            
            
        community_embeddings.append(embedding_node)
        
        # The embeddings will be saved in the txt file in a space-separated format, with each row representing the embedding vector of a specific node
        # np.savetxt() will save the file in the current working directory, which is the directory where your Python script or Jupyter Notebook is running

        np.savetxt(file_name, community_embeddings, delimiter=" ", fmt='%s')
        
    return community_embeddings



# Retrieving the community embedding for the healthy cell line

community_embeddings_h = community_embedding(partition_h, threshold_weights, communities_h, G_h, 'community_embeddings_h.txt')


# Retrieving the community embedding for the cancer cell line

community_embeddings_c = community_embedding(partition_c, threshold_weights, communities_c, G_c, 'community_embeddings_c.txt')



