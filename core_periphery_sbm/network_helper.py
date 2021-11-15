import numpy as np

def get_max_edges(block_r, block_s, block_ns):
    """
    Calculates the maximum number of edges that could possibly exist between two
    blocks

    Parameters
    ----------
    block_r, block_s: int
        Blocks to get the maximum number of edges between
    block_ns: 1D array
        Array counting the number of nodes in each block

    Returns
    -------
    M_rs: int
        The number of possible edges between block_r and block_s
    """
    if block_r == block_s:
        M_rs = block_ns[block_r]*(block_ns[block_r]-1)/2
    else:
        M_rs = block_ns[block_r]*block_ns[block_s]
    return M_rs

def get_ordered_block_stats(G, node_labels, n_blocks=None):
    """
    Calculates fundamental statistics for working with the block matrix of a
    network and then orders them according to the on-diagonal densities

    Parameters
    ----------
    G: NetworkX graph
        The graph for which to get block statistics
    node_labels: 1D array
        An array of the block label for each node in G. It is implicitly assumed
        that this array is sorted in the same way as sorted(G)
    n_blocks: int
        The number of blocks. If None, assumes max(node_labels)+1 is the number
        of blocks

    Returns
    -------
    block_ns: 1D array
        Array counting the number of nodes in each block
    block_ms: 2D array
        Matrix counting the number of edges that exist between pairs of blocks
    block_Ms: 2D array
        Matrix counting the maximum number of edges taht could potentially exist
        between pairs of blocks
    """
    block_ns,block_ms,block_Ms = get_block_stats(G, node_labels, n_blocks=n_blocks)
    return reorder_blocks(node_labels, block_ns, block_ms, block_Ms)

def get_block_stats(G, node_labels, n_blocks=None):
    """
    Calculates fundamental statistics for working with the block matrix of a
    network

    Parameters
    ----------
    G: NetworkX graph
        The graph for which to get block statistics
    node_labels: 1D array
        An array of the block label for each node in G. It is implicitly assumed
        that this array is sorted in the same way as sorted(G)
    n_blocks: int
        The number of blocks. If None, assumes max(node_labels)+1 is the number
        of blocks

    Returns
    -------
    block_ns: 1D array
        Array counting the number of nodes in each block
    block_ms: 2D array
        Matrix counting the number of edges that exist between pairs of blocks
    block_Ms: 2D array
        Matrix counting the maximum number of edges taht could potentially exist
        between pairs of blocks
    """
    # Get the number of blocks, 1 more than max node value because the blocks
    # are 0-indexed. Saves computational cost upfront if this is provided
    if n_blocks is None:
        n_blocks = max(node_labels)+1

    # Get the number of edges between layers
    seen_nodes = set()
    block_ns = np.zeros(n_blocks, dtype=int)
    block_ms = np.zeros((n_blocks,n_blocks), dtype=np.int64)
    for i,j in G.edges():
        i_block = node_labels[i]
        j_block = node_labels[j]
        # Update layer edge counts
        block_ms[i_block,j_block] += 1
        if i_block != j_block: # don't double count edge if in same layer
            block_ms[j_block,i_block] += 1
        # Update layer sizes
        if i not in seen_nodes:
            block_ns[i_block] += 1
            seen_nodes.add(i)
        if j not in seen_nodes:
            block_ns[j_block] += 1
            seen_nodes.add(j)

    # Get the maximum number of edges that could exist between blocks
    block_Ms = np.zeros((n_blocks,n_blocks), dtype=np.int64)
    for r in range(n_blocks):
        for s in range(r+1):
            M_rs = get_max_edges(r, s, block_ns)
            block_Ms[r,s] = M_rs
            if r != s:
                block_Ms[s,r] = M_rs

    return block_ns,block_ms,block_Ms

def get_layered_stats(block_ns, block_ms):
    """
    Collapses block edge counts down to layer edge counts

    Parameters
    ----------
    block_ns: 1D array
        Array counting the number of nodes in each block
    block_ms: 2D array
        Matrix counting the number of edges that exist between pairs of blocks

    Returns
    -------
    layer_ms: 1D array
        Array counting the number of edges that connect to each layer
    layer_Ms: 1D array
        Array counting the maximum number of edges that could potentially
        connect to each layer
    """
    # Get the number of blocks
    n_blocks = len(block_ns)

    # Collapse the layer edge counts down to layers
    layer_ms = np.zeros(n_blocks, dtype=np.int64)
    layer_Ms = np.zeros(n_blocks, dtype=np.int64)
    for r in range(n_blocks):
        for s in range(r+1):
            layer_ms[r] += block_ms[r,s]
            layer_Ms[r] += get_max_edges(r, s, block_ns)

    return layer_ms,layer_Ms

def get_on_diagonal_densities(block_ms, block_ns):
    """
    Calculates the densities within each block (densities of on diagonals of the
    block matrix)

    Parameters
    ----------
    block_ms: 2D array
        Matrix counting the number of edges that exist between pairs of blocks
    block_ns: 1D array
        Array counting the number of nodes in each block

    Returns
    -------
    ps: 1D array
        Array of the density within each block
    """
    ps = []
    for l in range(len(block_ns)):
        n = block_ns[l]
        m = block_ms[l,l]
        if n == 0 or n == 1:
            ps.append(0)
        else:
            ps.append(2 * m / (n * (n - 1)))
    return ps

def reorder_blocks(node_labels, block_ns, block_ms, block_Ms):
    """
    Sorts the fundamental block statistics according to the within block
    densities such that the highest density is indexed as 0, and so on

    Parameters
    ----------
    node_labels: 1D array
        An array of the block label for each node in G. It is implicitly assumed
        that this array is sorted in the same way as sorted(G)
    block_ns: 1D array
        Array counting the number of nodes in each block
    block_ms: 2D array
        Matrix counting the number of edges that exist between pairs of blocks
    block_Ms: 2D array
        Matrix counting the maximum number of edges taht could potentially exist
        between pairs of blocks

    Returns
    -------
    Returns the same parameters, but sorted according to within block densities
    """
    ps = get_on_diagonal_densities(block_ms, block_ns)
    ordered_blocks = np.argsort(ps)[::-1]
    old_block2new_block = {old_b:b for b,old_b in enumerate(ordered_blocks)}

    node_labels = [old_block2new_block[b] for b in node_labels]
    block_ns = [block_ns[b] for b in ordered_blocks]
    block_ms = block_ms[np.ix_(ordered_blocks,ordered_blocks)]
    block_Ms = block_Ms[np.ix_(ordered_blocks,ordered_blocks)]

    return node_labels,block_ns,block_ms,block_Ms
