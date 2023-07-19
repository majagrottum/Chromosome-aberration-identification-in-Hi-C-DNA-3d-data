# Visualization of clusters in network graphs

## About this library

This code was used in a project for indentifying chromosome aberrations in Hi-C DNA data, but can also be used for different networks. 
Using the functions in this code you can create a graph from data (in the form of an adjacency matrix) saved in a .txt, remove the isolated nodes, and then retrieve the calculated node embeddings using the node2vec algorithm.
Furthermore can these embeddings be used to find clusters within the network using the HDBSCAN algorithm. 
Then, using Principal Component Analysis (PCA) the node embeddings can be visualized in 2D (or any other dimension specified with the argument n_components) by reducing the dimensions of the data.
The data is then visualized in a plot through the principal components, assigning a different color to each cluster in the network. 
In the case of the Hi-C data, the visualization of the clusters was obtained to identify chromosome aberrations in a cancer cell line, meaning that some chromosomes would be separated into different clusters.

The different parameters/arguments for the different functions are specified in more detail throughout the code in the library.py file.

## Installation

To install this library, clone the repository and use pip by opening your terminal or command prompt and run the following commands:

```
git clone https://github.com/majagrottum/Visualization-of-clusters-in-network-graphs
cd Visualization-of-clusters-in-network-graphs
pip install -r requirements.txt

```
  
