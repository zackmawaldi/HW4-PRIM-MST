import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:

            if len(adjacency_mat.shape) != 2:
                raise ValueError('Adjacency matrix must be 2D array')
            
            if adjacency_mat.shape[0] != adjacency_mat.shape[1]:
                raise ValueError('Adjacency 2D array must be even in both dimentions')
            
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None


    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')


    def size(self):
        return self.adj_mat.shape[0]


    def get_nodes(self):
        return list(range(len(self.adj_mat)))


    def neighbors(self, node):
        # assumes weights can not be 0...
        return [i for i, v in enumerate(self.adj_mat[node]) if v != 0]

    
    def add_node(self, node):
        current_size = self.size()
        if node >= current_size:
            # calculate the new size needed, which is node + 1
            new_size = node + 1
            # expand the adjacency matrix to the new size, init values to 0
            self.adj_mat = np.pad(self.adj_mat, 
                                ((0, new_size - current_size), 
                                (0, new_size - current_size)), 
                                mode='constant', constant_values=0)
        else:
            # if node < current_size, then node is alread in the graph, thus
            # at least for our use case, leave as is.
            pass



    def add_edge(self, node1, node2, weight=1):
        if node1 < self.size() and node2 < self.size():
            self.adj_mat[node1][node2] = weight
            self.adj_mat[node2][node1] = weight
        else:
            raise ValueError("Node1 and Node2 should be within the current size of the graph.")


    def get_edge_value(self, node1, node2):
        if node1 < self.size() and node2 < self.size():
            return self.adj_mat[node1][node2]
        else:
            raise ValueError("Node1 and Node2 should be within the current size of the graph.")
    

    def edges_count(self):
        count = 0
        for i in range(self.size()):
            for j in range(i + 1, self.size()):
                if self.adj_mat[i][j] != 0:
                    count += 1
        return count
                

    def construct_mst(self):
        """
        Constructs the Minimum Spanning Tree (MST) of the graph using Prim's algorithm.

        This method initializes an empty graph for the MST and starts with a randomly selected first node.
        It then continuously adds the smallest edge connecting a visited node to an unvisited node, which should avoid cycles.
        Repeats until all nodes are included (accept), or no more edges can be added (reject).
        The resulting MST is stored in self.mst. If rejected, self.mst == None.
        """
    
        # initialize a copy graph to be our final output subgraph with all 0's
        empty_graph_matrix = np.zeros_like(self.adj_mat)
        mst_graph = Graph(empty_graph_matrix)

        nodes_array = np.array(self.get_nodes())
        first_node = np.random.choice(nodes_array)

        visited = {first_node} # {} == set, not dict

        # get edges list of first_node in format [(w, first_node, node_to), ... etc]
        edges_frontier = [(self.get_edge_value(first_node, neighbor), first_node, neighbor) for neighbor in self.neighbors(first_node)]
        
        # cast the list of tuple edges into a heap
        # tuples are compared element-wise, thus this should work
        heapq.heapify(edges_frontier)

        while len(visited) < self.size():
            if not edges_frontier:
                # if no more edges, then graph must be not connected, thus no mst
                self.mst = None
                return

            weight, node_from, node_to = heapq.heappop(edges_frontier) # pop = smallest edge
            if node_to not in visited:
                visited.add(node_to)
                mst_graph.add_node(node_to)
                mst_graph.add_edge(node_from, node_to, weight)

                # expand edge frontier based on just-added edge
                for neighbor in self.neighbors(node_to):
                    if neighbor not in visited:

                        # for the edge below, 'node_to' here is a misnomer
                        edge = (self.get_edge_value(node_to, neighbor), node_to, neighbor)
                        heapq.heappush(edges_frontier, edge)

        self.mst = mst_graph.adj_mat