import numpy as np
import networkx as nx
import numpy.typing as npt

def calc_network_density(weighting_matrix,N):
    """
    Print network density given by actual_connections / potential_connections

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    actual_connections = weighting_matrix.sum()
    potential_connections = (N * (N - 1))
    network_density = actual_connections / potential_connections
    return network_density

def create_weighting_matrix(N,K,prob_rewire,set_seed) -> tuple[npt.NDArray, npt.NDArray, nx.Graph]:
    """
    Create watts-strogatz small world graph using Networkx library

    Parameters
    ----------
    None

    Returns
    -------
    weighting_matrix: npt.NDArray[bool]
        adjacency matrix, array giving social network structure where 1 represents a connection between agents and 0 no connection. It is symetric about the diagonal
    norm_weighting_matrix: npt.NDArray[float]
        an NxN array how how much each agent values the opinion of their neighbour. Note that is it not symetric and agent i doesn't need to value the
        opinion of agent j as much as j does i's opinion
    ws: nx.Graph
        a networkx watts strogatz small world graph
    """
    ws = nx.watts_strogatz_graph(
        n=N, k=K, p=prob_rewire, seed=set_seed
    )  # Watts–Strogatz small-world graph,watts_strogatz_graph( n, k, p[, seed])
    weighting_matrix = nx.to_numpy_array(ws)

    return (
        weighting_matrix,
        ws,
    )

if __name__ == "__main__":
    N = 100
    prob_rewire = 0.1
    set_seed = 1
    """
    for k in range(50):
        weighting_matrix, ws = create_weighting_matrix(N,k,prob_rewire,set_seed)
        density = calc_network_density(weighting_matrix,N)
        print("k:",k,"k/2(N-1)",k/(N-1),"2mk/N(N-1):",(N*k/2)/(N*(N-1)/2), "networkx density", nx.density(ws))
    """
    
    for p in np.linspace(0.01, 0.5):
        k = round(p*(N-1))
        ws = nx.watts_strogatz_graph(n=N, k=k, p=prob_rewire, seed=set_seed) 
        print("p = ", p,"k = ", k,"density =", nx.density(ws))
