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



def test_symmetricity(filename, splitting):

  """This function tests if the output graph from the create_graph function is symmetric.
  If the graph is undirected and the adjacency matrix is symmetric, the graph edges should be symmetric as well."""

  graph = create_graph(filename, splitting)

  # If the condition evaluates to True, it means the graph is symmetric (undirected and has the same structure as its undirected version).
  assert nx.is_isomorphic(graph, graph.to_undirected()), "Test failed: The graph should be symmetric."





def test_remove_isolated_nodes():

  """This function tests if the remove_isolated_nodes function works correctly in the case of a graph with one isolated node."""

  graph = nx.Graph()
  
  graph.add_nodes_from([1, 2, 3])
  
  graph.add_edges_from([(1, 2)])
  
  remove_isolated_nodes(graph)
  
  assert graph.number_of_nodes() == 2, "Test failed: The graph should have 2 nodes after removing isolated nodes."
  
  assert graph.number_of_edges() == 1, "Test failed: The graph should have 1 edge after removing isolated nodes."



def test_all_nodes_isolated():

  """This function tests if the remove_isolated_nodes function makes an empty graph when all the nodes in the graph are isolated."""

  graph = nx.Graph()
  
  graph.add_nodes_from([1, 2, 3])
  
  remove_isolated_nodes(graph)
  
  assert graph.number_of_nodes() == 0, "Test failed: The graph should have 0 nodes after removing isolated nodes."
  
  assert graph.number_of_edges() == 0, "Test failed: The graph should have 0 edges after removing isolated nodes."



def test_no_isolated_nodes():

  """This function tests if the remove_isolated_nodes function leaves the graph unchanged when there are no isolated nodes."""

  graph = nx.Graph()
  
  graph.add_nodes_from([1, 2, 3])
  
  graph.add_edges_from([(1, 2), (2, 3)])
  
  remove_isolated_nodes(graph)
  
  assert graph.number_of_nodes() == 3, "Test failed: The graph should have 3 nodes as there are no isolated nodes."
  
  assert graph.number_of_edges() == 2, "Test failed: The graph should have 2 edges as there are no isolated nodes."





def test_node_embeddings():

def test_embedding_list():

def test_clustering_HDBSCAN():

def test_principal_component_analysis():
