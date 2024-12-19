"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *
from src.operations import *

def plot_graph(G:nx.Graph, nd_values:np.ndarray, pos:Optional[dict]=None, cmap:Optional[colors.Colormap]=None, 
               scale:int=100, ax:matplotlib.axes.Axes=None, scolor:Optional[list]=["red", "blue"], 
               colorbar:bool=False, nodetype:bool="size", **kwds):
    """
    Visualize a signal on a directed graph.

    Plots a directed graph with node size and/or color determined by node values.
    Node size is scaled by the 'scale' parameter to be visible.
    Node color is determined by the sign of the node value (positive or negative)
    if a color map is not provided. If a color map is provided, node color 
    is mapped to the normalized node value.

    Parameters
    ----------
    G : networkx.Graph
        Directed graph to plot

    nd_values : numpy.ndarray
        Node values, used for size and/or color

    pos : dict, optional
        Node positions for graph layout

    cmap : matplotlib.colors.Colormap, optional
        Color map to use for node colors
    
    scale : float, optional
        Scaling factor for node sizes

    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    scolor : list, optional
        Default node colors if cmap not provided

    colorbar : bool, optional
        Whether to draw a colorbar (requires cmap)

    nodetype : str
        - "color" colors is showing the difference between nodes values
        - "size" size of nodes is showing the difference between nodes values

    Returns
    -------
    None
    
    """
    if cmap is None:
        nd_color = [scolor[0] if nd > 0 else scolor[1] for nd in nd_values]
    else:
        normalized_values = nd_values - nd_values.min()
        normalized_values /= normalized_values.max()

        nd_color = [cmap(normalized_values[k]) for k in range(len(normalized_values))]

    node_values = scale * np.abs(nd_values)
    if nodetype == "color":
        nx.draw(G,arrows=True,node_color=nd_values,
                pos=pos,ax=ax,cmap=cmap, **kwds)
    elif nodetype == "size":
        nx.draw(G,arrows=True,node_size=node_values,node_color=nd_color,
                pos=pos,ax=ax,cmap=cmap, **kwds)
    else:
        print("Unsupported input ... plotting nodes with default size and color")

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm)

def diagonalize(arr:np.ndarray, laplacian:bool=False):
    """
    Diagonalize the given adjacency matrix by calling an external
    MATLAB script.

    This function converts the adjacency matrix to a MATLAB compatible 
    format, saves it to a temporary file, calls the MATLAB script to
    diagonalize it, and returns the diagonalized matrix.

    Parameters
    ----------
    arr : numpy.ndarray
        The adjacency matrix to diagonalize.

    laplacian : bool, optional
        Whether the matrix should be converted to a Laplacian matrix
        before diagonalizing. Default is False.

    Returns
    ------- 
    numpy.ndarray
        The diagonalized adjacency matrix.
    """

    def adjcencymat2matlab_digraph(A):
        """
        Compute adjacency format for matlab compatible from numpy array
        """
        weight = []
        tmp = np.where(A != 0)
        for k in range(len(tmp[0])):
            x = tmp[0][k]
            y = tmp[1][k]
            weight.append(A[x, y])
        ret = {"D1": tmp[0] + 1, "D2": tmp[1] + 1, "weight": np.array(weight)}
        return ret

    # Generate matlab format
    sio.savemat("/tmp/tmp.mat", adjcencymat2matlab_digraph(arr))

    if laplacian:
        assert os.path.exists(
            "../outsource/digraphSP-generalized-boundaries/diagonalize.m"
        ), "matlab diagonalizing script not found"

        os.system(
            "/Applications/MATLAB_R2019b.app/bin/matlab -maci64 -nodisplay -nosplash -nodesktop -r \"run('../outsource/digraphSP-generalized-boundaries/diagonalize_laplacian.m');exit;\" | tail -n +11"
        )

        A = sio.loadmat("../outsource/digraphSP-generalized-boundaries/MNoJordan.mat")[
            "MNoJordanBlocks"
        ]
        return A

    else:
        assert os.path.exists(
            "../outsource/digraphSP-generalized-boundaries/diagonalize.m"
        ), "matlab diagonalizing script not found"

        # Call diagonalization of the script
        os.system(
            "/Applications/MATLAB_R2019b.app/bin/matlab -maci64 -nodisplay -nosplash -nodesktop -r \"run('../outsource/digraphSP-generalized-boundaries/diagonalize.m');exit;\" | tail -n +11"
        )

        A = sio.loadmat("../outsource/digraphSP-generalized-boundaries/MNoJordan.mat")[
            "MNoJordanBlocks"
        ]
        return A

def prep_transform(A:np.ndarray, gso:str="adj", composite:bool=True, verbose:bool=False, 
                   in_degree:bool=True):
    """
    Prepare the matrices needed for all transforms
    with A being the adjacency matrix.

    Computes the graph Laplacian L, eigenvector matrices U and V,
    inverse Uinv, diagonal matrix of eigenvalues S, identity matrix I,
    inverse Sinv.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix
    gso : str
        The generalized eigendecomposition to use. 
        Can be "adj" for adjacency or "laplacian" for graph laplacian.
    composite : bool
        Whether to compute the composite transform matrices.
    verbose : bool
        Whether to print debug information.

    Returns
    -------
    L : numpy.ndarray
        The graph Laplacian.
    U : numpy.ndarray
        The left eigenvector matrix.
    V : numpy.ndarray
        The right eigenvector matrix.  
    Uinv : numpy.ndarray
        The inverse of the left eigenvector matrix.
    S : numpy.ndarray
        The diagonal matrix of eigenvalues. Only returned if composite=True.
    J : numpy.ndarray 
        The identity matrix. Only returned if composite=True.
    Sinv : numpy.ndarray
        The inverse of the diagonal eigenvalue matrix. Only returned if composite=True.
    """

    if gso == "adj":
        L = deepcopy(A)
    elif gso == "laplacian":
        L = compute_directed_laplacian(A, in_degree=in_degree)
    else:
        print("Unsupported GSO ... using adjacency matrix")
        L = deepcopy(A)

    U, V = compute_basis(L, verbose=verbose, gso=gso)
    Uinv = np.linalg.inv(U)

    # Clean all 1e-10 real or imaginary by mapping to 0
    cleanV = V.real * (np.abs(V.real) > 1e-10) + 1j * V.imag * (np.abs(V.imag) > 1e-10)

    if composite:
        S, I = compute_basis(hermitian(np.diag(cleanV)) @ hermitian(U) @ U @ np.diag(cleanV))
        J = np.diag(I)
        Sinv = np.linalg.inv(S)

        return L, U, cleanV, Uinv, S, J, Sinv
    else:
        return L, U, cleanV, Uinv, None, None, None

def visual_graph(A:np.ndarray, diracsig:Optional[list]=None, 
              pos:Optional[dict]=None, layout:Optional[dict]=None, figsize:tuple=(5, 5)):
    """
    Visualize a graph based on its adjacency matrix.
    """
    N = len(A)
    _, ax = plt.subplots(1, figsize=figsize)
    G = nx.from_numpy_array(A.T, create_using=nx.MultiDiGraph())
    if (pos is None) and (layout is None):
        pos = nx.kamada_kawai_layout(G)
    else:
        if pos is None:
            pos = layout(G)

    sig = np.zeros(len(A))
    if diracsig is None:
        sig = np.ones(N)
    else:
        if type(diracsig) == list:
            sig = np.zeros(N)
            for ds in diracsig:
                sig[ds] = 1.0
        elif type(diracsig) == int:
            sig = np.zeros(N)
            if diracsig != -1:
                sig[diracsig] = 1.0

    plot_graph(G, sig, pos, ax=ax, edgecolors='black')
    plt.show()

    return G, pos
    

def vis_graph(A:np.ndarray, U:Optional[np.ndarray]=None, diracsig:Optional[list]=None, 
              pos:Optional[dict]=None, layout:Optional[dict]=None, figsize:tuple=(9, 3)):
    """
    Visualize a graph based on its adjacency matrix.
    NOTE: Careful -> we transpose the adjacency to fix the convention of
    "Graph Fourier Transform Based on Directed Laplacian" and fit the displaying of networkx
    w_{i,j} is the edge that goes from j to i

    Plots the absolute value of the eigenvector matrix U multiplied 
    by its conjugate transpose. This shows the frequency structure of the 
    graph.

    Also visualizes the graph topology using networkx, with optional
    node signal diracsig.

    Parameters
    ----------
    A : numpy.ndarray
        The adjacency matrix.
    U : numpy.ndarray, optional
        The eigenvector matrix.
    diracsig : numpy.ndarray or list or int, optional
        The node signal, defaults to random noise.
        If a list, the indices are nodes with dirac signals.
        If an int, the node index with a dirac signal.
        If -1, no dirac signal.
    pos : dict, optional
        The node positions for visualization.
    layout : function, optional
        The networkx layout function.

    Returns
    -------
    G : networkx.MultiDiGraph
        The graph object.
    pos : dict
        The node positions.
    """
    N = len(A)
    if U is None:
        _, ax = plt.subplots(1, figsize=figsize)
        G = nx.from_numpy_array(A.T, create_using=nx.MultiDiGraph())
        if (pos is None) and (layout is None):
            pos = nx.kamada_kawai_layout(G)
        else:
            if pos is None:
                pos = layout(G)

        sig = np.zeros(len(A))
        if diracsig is None:
            sig = np.abs(np.random.random(N))
        else:
            if type(diracsig) == list:
                sig = np.zeros(N)
                for ds in diracsig:
                    sig[ds] = 1.0
            elif type(diracsig) == int:
                sig = np.zeros(N)
                if diracsig != -1:
                    sig[diracsig] = 1.0


        plot_graph(G, sig, pos, ax=ax)
        plt.show()

        return G, pos
    
    else:
        _, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(np.abs(hermitian(U) @ U))

        G = nx.from_numpy_array(A.T, create_using=nx.MultiDiGraph())
        if (pos is None) and (layout is None):
            pos = nx.kamada_kawai_layout(G)
        else:
            if pos is None:
                pos = layout(G)

        sig = None
        if diracsig is None:
            sig = np.abs(np.random.random(N))
        else:
            if type(diracsig) == list:
                sig = np.zeros(N)
                for ds in diracsig:
                    sig[ds] = 1.0
            elif type(diracsig) == int:
                sig = np.zeros(N)
                if diracsig != -1:
                    sig[diracsig] = 1.0

        plot_graph(G, sig, pos, ax=ax[1])
        plt.show()

        return G, pos

def make_graph(N:int, graph_type:str, split:bool=False):
    """
    Generate Adjacency matrix of a graph of N nodes.

    Supported graph types are:
    - line
    - cycle
    - bicycle
    - tricycle

    Parameters
    ----------
    N : int
        Number of nodes in graph
    graph_type : str
        Type of graph to generate. Options are "line", "cycle", "bicycle", "tricycle".
    split : bool, optional
        Whether to split the bicycle graph into two components. Default is False.

    Returns
    -------
    A : numpy.ndarray
        The generated adjacency matrix for the graph.

    """

    if graph_type == "line":
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        A = np.concatenate([bound, A], axis=1)

    elif graph_type == "cycle":
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        bound[-1] = 1.0
        A = np.concatenate([bound, A], axis=1)

    elif graph_type == "bicycle":
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        bound[-1] = 1.0
        A = np.concatenate([bound, A], axis=1)

        # Adding one sub cycle
        if N <= 12 or split == True:
            A[N // 2, 0] = 1
        else:
            A[4 * N // 6, 5 * N // 6] = 1

    elif graph_type == "tricycle":
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        bound[-1] = 1.0
        A = np.concatenate([bound, A], axis=1)

        # Adding two sub cycle
        A[N // 6, 2 * N // 6] = 1
        A[4 * N // 6, 5 * N // 6] = 1

    else:
        print("Not supported format : use either cycle / bicycle / tricycle")
        raise IndexError

    return A

def combine_graphs(A: np.ndarray, B: np.ndarray, nodes_listA: list, nodes_listB: list):
    """
    Combine graphs by union and adding edges between corresponding nodes.
    Elements in nodes_listA and nodes_listB are indices of nodes to connect 
    between graphs A and B respectively. Negative indexes refer to going from 
    B to A while positive indexes refer to going from A to B.

    Parameters
    ----------
    A : np.ndarray
        Graph A adjacency matrix
    B : np.ndarray
        Graph B adjacency matrix
    
    nodes_listA : list
        Nodes in A to connect
    nodes_listB : list    
        Nodes in B to connect

    Returns
    -------
    ret : np.ndarray
        Combined graph adjacency matrix
    """

    a = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph)
    b = nx.convert_matrix.from_numpy_array(B, create_using=nx.DiGraph)

    c = nx.union(a, b, rename=("a-", "b-"))
    for k in range(len(nodes_listA)):
        nA, nB = nodes_listA[k], nodes_listB[k]
        if nA < 0 and nB < 0:
            c.add_edge(f"b-{-nB}", f"a-{-nA}")
        else:
            c.add_edge(f"a-{nA}", f"b-{nB}")

    ret = np.array(nx.adjacency_matrix(c).todense())
    return ret

def get_cycles(G: nx.Graph, start_idx: int, max_depth: int, verbose:bool=True):
    """
    Find all cycles reachable from a start node within a given maximum depth.

    Parameters
    ----------
    G : networkx.Graph
        The graph to search for cycles.
    start_idx : int 
        The index of the node to start the search from.
    max_depth : int
        The maximum depth to search for cycles.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    unique_cycles : list
        A list of lists, where each inner list represents a cycle path.
    """
    from collections import Counter

    def findPaths(G, u, n):
        if n == 0:
            return [[u]]
        # paths = [ [u] + path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        paths = [
            [u] + path
            for neighbor in G.neighbors(u)
            for path in findPaths(G, neighbor, n - 1)
        ]
        return paths

    allpaths = findPaths(G, start_idx, max_depth)

    # 1. Search for cycles
    if verbose: print(f"Finding cycles up to depth {max_depth} from node {start_idx}...")
    paths_with_cycles = np.where(np.sum(np.array(allpaths) == start_idx, axis=1) > 1)[0]
    paths_with_cycles = np.array(allpaths)[paths_with_cycles]

    # 2. Trim the sequences to only keep the cycles
    if verbose: print(f"Trimming paths to isolate cycles...")
    trimed_paths = []
    for k in range(len(paths_with_cycles)):
        cstart, cend = np.where(paths_with_cycles[k] == start_idx)[0][[0, 1]]
        sequence = paths_with_cycles[k][cstart : cend + 1]

        if (np.array(list(Counter(sequence).values())) > 1).sum() == 1:
            trimed_paths.append(sequence)

    # 3. Remove repeating sequences
    if verbose: print("Removing repeating cycles...")
    unique_cycles = []
    add_flag = True
    for p in trimed_paths:
        for cur in unique_cycles:
            if np.any(p == cur):
                add_flag = False
        if add_flag:
            unique_cycles.append(p)
        add_flag = True

    # 4. Verify that all inputs are indeed cycles and remove the last value to close the loop
    if verbose: print("Verifying cycles and closing loops...")
    unique_cycles = [p[:-1] for p in unique_cycles if p[0] == p[-1]]
        
    return unique_cycles

def var_generator(A:np.ndarray, active_nodes:list, amplitude_nodes:list, 
                  time_nodes:list, n_iter:int,
                  add_noise:bool, time_noise:list, 
                  gamma:float=1, seed:int=99):
    """
    Generates a sequence of directed graph signals over time using a graph spreading process.

    Parameters
    ----------
        A (numpy.ndarray): The adjacency matrix of the graph.
        active_nodes (list): A list of indices of the active nodes in the graph.
        amplitude_nodes (list): A list of amplitudes to be applied to the active nodes.
        time_nodes (list): A list of time steps at which the active node amplitudes should be applied.
        n_iter (int): The number of time steps to simulate.
        add_noise (bool): Whether to add Gaussian noise to the graph signals.
        time_noise (list): A list of time steps at which Gaussian noise should be added.
        gamma (float, optional): A scaling factor for the adjacency matrix. Defaults to 1.
        seed (int, optional): A seed for the random number generator. Defaults to 99.

    Returns
    -------
        directed_logs (numpy.ndarray): A 2D array of shape (n_iter, graphdim) containing the sequence of directed graph signals.
    """
    np.random.seed(seed)

    graphdim = len(A)

    # Initial condition: Implemented to be Gaussian Impulse
    initial_cond = np.random.normal(0, 1, graphdim)
    initial_directed = deepcopy(initial_cond)
    directed_logs = [initial_directed]

    # Defining GSO
    muA = gamma * A

    # Generating the diffusion processes
    for _iter in range(n_iter - 1):

        if (_iter in time_noise) and add_noise:
            source_random = np.random.normal(0, 1, graphdim)
        else:
            source_random = np.zeros(graphdim)

        # Spreading process
        initial_directed = muA @ directed_logs[-1]

        # Node Inherent process
        initial_directed += source_random
        if _iter in time_nodes:
            for lidx, l in enumerate(active_nodes):
                initial_directed[l] += amplitude_nodes[lidx]

        directed_logs.append(initial_directed)

    directed_logs = np.array(directed_logs)
    return directed_logs

def plot_spectrum_gft(signals:np.ndarray, U:np.ndarray, V:np.ndarray, Uinv:Optional[np.ndarray]=None, labels:Optional[list]=None,
                       spectreonly:bool=False, figsize:tuple=(9, 3), plot_real:bool=True):
    """
    Plot the graph Fourier transform spectrum of the given signals.

    Parameters
    ----------
    signals : array-like
        The signals to plot the GFT spectrum for.
    U : array-like
        The eigenvectors of the graph Laplacian.  
    V : array-like
        The eigenvalues of the graph Laplacian.
    Uinv : array-like, optional
        The inverse of the eigenvectors of the graph Laplacian.  
    labels : list of str, optional
        Labels for each signal to include in the legend.
    spectreonly : bool, optional
        Whether to only plot the spectrum without the coefficient positions. Default is False.
    figsize : tuple, optional  
        The figure size. Default is (9, 3).
    plot_real : bool, optional
        Whether to double plot the real eigenvalues. Default is True.
        
    Returns
    -------
    None
    """

    signals = np.asarray(signals)
    cm = plt.get_cmap("gist_rainbow")
    if len(signals.shape) == 1:
        signals = [signals]
        colors = ["k"]
    else:
        colors = [cm(1.0 * i / len(signals)) for i in range(len(signals))]
    if not (labels is None):
        if len(labels) != len(signals):
            raise ValueError("Length of labels and numbers of signals are different")

    if spectreonly:
        _, ax = plt.subplots(1, 2, figsize=figsize)
        real_freqs = np.abs(V.imag) < 1e-10
        for sidx, signal in enumerate(signals):
            stilt = GFT(signal, U, Uinv=Uinv)

            # Compute frequencies and associated magnitudes
            freqs = []
            mags = []
            for freqnb in range(len(stilt)):
                # Add both the positive part and the negative part
                if (real_freqs[freqnb] == 1) and plot_real:
                    freqs.append(-np.abs(V[freqnb]))
                    freqs.append(np.abs(V[freqnb]))
                    mags.append(np.abs(stilt[freqnb]))
                    mags.append(np.abs(stilt[freqnb]))
                else:
                    freqs.append(np.sign(V.imag[freqnb]) * np.abs(V[freqnb]))
                    mags.append(np.abs(stilt[freqnb]))
            freqs = np.asarray(freqs)
            mags = np.asarray(mags)
            if not (labels is None):
                ax[0].plot(
                    freqs[np.argsort(freqs)],
                    mags[np.argsort(freqs)],
                    label=f"power {labels[sidx]}",
                    c=colors[sidx],
                )
                ax[1].plot(
                    np.arange(len(stilt)),
                    np.cumsum(np.abs(stilt) / np.abs(stilt).sum()),
                    label=f"power {labels[sidx]}",
                    c=colors[sidx],
                )
            else:
                ax[0].plot(
                    freqs[np.argsort(freqs)],
                    mags[np.argsort(freqs)],
                    # label=f"power signal{sidx+1}",
                    c=colors[sidx],
                )
                ax[1].plot(
                    np.arange(len(stilt)),
                    np.cumsum(np.abs(stilt) / np.abs(stilt).sum()),
                    # label=f"power signal{sidx+1}",
                    c=colors[sidx],
                )

            ax[1].set_xlabel("Frequencies")
            ax[1].set_ylabel("Power")

            ax[0].set_xlabel("TV-proxy")
            ax[0].set_ylabel("Power")
            if not (labels is None):
                ax[1].legend(prop={"size": 9})
                ax[0].legend(prop={"size": 9})
        plt.show()
    else:
        _, ax = plt.subplots(1, 2, figsize=figsize)
        real_freqs = np.abs(V.imag) < 1e-10
        for sidx, signal in enumerate(signals):
            stilt = GFT(signal, U, Uinv=Uinv)
            
            # Compute frequencies and associated magnitudes
            freqs = []
            mags = []
            for freqnb in range(len(stilt)):
                # Add both the positive part and the negative part
                if (real_freqs[freqnb] == 1) and plot_real:
                    freqs.append(-np.abs(V[freqnb]))
                    freqs.append(np.abs(V[freqnb]))
                    mags.append(np.abs(stilt[freqnb]))
                    mags.append(np.abs(stilt[freqnb]))
                else:
                    freqs.append(np.sign(V.imag[freqnb]) * np.abs(V[freqnb]))
                    mags.append(np.abs(stilt[freqnb]))
            freqs = np.asarray(freqs)
            mags = np.asarray(mags)

            if not (labels is None):
                ax[0].scatter(
                    stilt.real,
                    stilt.imag,
                    label=f"coef position {labels[sidx]}",
                    c=colors[sidx],
                )
                ax[1].plot(
                    freqs[np.argsort(freqs)],
                    mags[np.argsort(freqs)],
                    label=f"coef power {labels[sidx]}",
                    c=colors[sidx],
                )
            else:
                ax[0].scatter(
                    stilt.real,
                    stilt.imag,
                    # label=f"coef position signal{sidx+1}",
                    c=colors[sidx],
                )
                ax[1].plot(
                    freqs[np.argsort(freqs)],
                    mags[np.argsort(freqs)],
                    # label=f"coef power signal{sidx+1}",
                    c=colors[sidx],
                )

            ax[0].set_xlabel("Real-part")
            ax[0].set_ylabel("Imaginary-part")

            ax[1].set_xlabel("TV-proxy")
            ax[1].set_ylabel("Power")
            if not (labels is None):
                ax[0].legend(prop={"size": 9})
                ax[1].legend(prop={"size": 9})
        plt.show()