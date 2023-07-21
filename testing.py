def test_create_graph_output_type(filename, splitting):

  """This function tests if the output of the create_graph function is an undirected graph or a subclass of nx.Graph."""

  graph = create_graph(filename, splitting)
  
  assert isinstance(graph, nx.Graph), "Test failed: The output should be an instance of nx.Graph."


def test_nodes_and_edges(filename, splitting):

  """This function tests if the output graph from the create_graph function have at least one node and a non-negative number of edges."""

  graph = create_graph(filename, splitting)

  assert graph.number_of_nodes() > 0, "Test failed: The graph should have at least one node."
  
  assert graph.number_of_edges() >= 0, "Test failed: The graph should have a non-negative number of edges."


def test_contain_all_nodes(filename, splitting):

  """This function tests if the output graph from the create_graph function contains all nodes from the adjacency matrix."""

  adjacency_matrix = np.loadtxt(filename, delimiter=splitting)
  
  expected_nodes = range(len(adjacency_matrix))

  graph = create_graph(filename, splitting)
  
  assert set(graph.nodes()) == set(expected_nodes), "Test failed: The graph should contain all nodes from the adjacency matrix."


def test_edge_weights(filename, splitting):

  """This function tests if the corresponding graph edges in the output graph from the create_graph function 
  have the same weights as the distances between nodes in the adjacency matrix."""

  adjacency_matrix = np.loadtxt(filename, delimiter = splitting)
  
  graph = create_graph(filename, splitting)

  for u, v in graph.edges():
    
    if "weight" in graph[u][v]:
      
      weight = graph[u][v]["weight"]

    else:
      
      # Default weight is set to 1 if not explicitly provided
      weight = 1
          
    assert weight == adjacency_matrix[u][v], "Test failed: Edge weight between nodes {u} and {v} should be {adjacency_matrix[u][v]}."






def test_remove_isolated_nodes():

def test_node_embeddings():

def test_embedding_list():

def test_clustering_HDBSCAN():

def test_principal_component_analysis():
