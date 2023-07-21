def test_create_graph_output_type(filename, splitting):

  """This function tests if the output of the create_graph function is an undirected graph or a subclass of nx.Graph."""

  graph = create_graph(filename, splitting)
  
  assert isinstance(graph, nx.Graph), "Test failed: The output should be an instance of nx.Graph."


def test_nodes_and_edges(filename, splitting):

  """This function tests if the output graph have at least one node and a non-negative number of edges."""

  graph = create_graph(filename, splitting)

  assert graph.number_of_nodes() > 0, "Test failed: The graph should have at least one node."
  
  assert graph.number_of_edges() >= 0, "Test failed: The graph should have a non-negative number of edges."

def test_remove_isolated_nodes():

def test_node_embeddings():

def test_embedding_list():

def test_clustering_HDBSCAN():

def test_principal_component_analysis():
