import os
import sys
import warnings
import numpy as np
import networkx as nx
from collections import Counter

from . import inference as infh
from . import network_helper as nh

# ------------------------------------------------------------------------------
# ------------------------ General core-periphery model ------------------------
# ------------------------------------------------------------------------------
class CorePeriphery(object):
    def __init__(self, n_blocks=2, n_gibbs=200, n_mcmc=None, eps=1,
                 mcmc_init="random", moves="random", seed=None):
        """
        Parameters
        ----------
        n_blocks: int
            The number of blocks to infer for the core-periphery
        n_gibbs: int
            The number of steps to use in the Gibbs sampling between the ps and
            the node labels
        n_mcmc: int
            The number of steps to use in the MCMC sampling of the node labels
            within the Gibbs sampling. Defaults to n_nodes * n_blocks
        eps: float
            Parameter for neighbor-based MCMC proposals that is relatively
            arbitrary. As eps goes to infinity, the proposed moves become
            equivalent to choosing uniformaly at random from all blocks. Does
            not matter if moves="random". See `moves` parameter below.
        mcmc_init: str
            If "previous", then initialize each MCMC chain from the final
            state of the previous chain. If "random", initialize each chain
            randomly. The first initialization is always random.
        moves: str
            If "random", label proposals are drawn uniformly at random among all
            blocks during MCMC. If "neighbor", labels are proposed according to
            the neighbor labels. For details: Peixoto, T. P. (2014). Efficient
            Monte Carlo and greedy heuristic for the inference of stochastic
            block models. *Physical Review E*.
        seed: int
            Seed passed to np.random.seed()
        """
        # Initialize inference parameters
        self.n_blocks = n_blocks
        self.blocks = list(range(n_blocks))
        self.n_gibbs = n_gibbs
        self.n_mcmc = n_mcmc
        self.eps = eps
        self.T = 0
        # Sampling parameters
        if (moves != 'random') and (moves != 'neighbor'):
            warning = 'Label move proposals not specified correctly,'\
                      +' defaulting to "random".'
            warnings.warn(warning)
            self.moves = 'random'
        else:
            self.moves = moves
        if (mcmc_init != 'random') and (mcmc_init != 'previous'):
            warning = 'MCMC initialization not specified correctly, defaulting'\
                      +' to "previous".'
            warnings.warn(warning)
            self.mcmc_init = 'previous'
        else:
            self.mcmc_init = mcmc_init
        # Initialize data structures for inference
        self.block_ns = np.zeros((self.n_gibbs, self.n_blocks), dtype=int)
        self.block_ms = np.zeros((self.n_gibbs, self.n_blocks, self.n_blocks),
                                  dtype=np.int64)
        self.block_Ms = np.zeros((self.n_gibbs, self.n_blocks, self.n_blocks),
                                  dtype=np.int64)
        self.log_posteriors = np.zeros(self.n_gibbs)
        self.log_labels_given_ps = np.zeros(self.n_gibbs)
        self.accepts = np.zeros(self.n_gibbs-1)

        if seed is not None:
            np.random.seed(seed)

    def infer(self, G):
        """
        Infer the core-periphery structure of a network

        Parameters
        ----------
        G: NetworkX graph
            Graph for which to infer a core-periphery structure
        """
        # Clean and validate network before continuing
        H = self.clean_network(G)

        # Initialize the inference, implicitly sets block statistics
        init = self.initialize_inference(H)
        if init is False:
            warning = 'Random initialization of nodes resulted in blocks that'\
                      +' were empty or disconnected. You likely have too many'\
                      +' blocks for the size of your network.'\
                      +'\n\nStopping inference.'
            warnings.warn(warning)
            return False

        # Run the Gibbs sampling over ps and labels
        for T in range(1, self.n_gibbs):
            self.T = T
            self.gibbs_sample_labels(H)
            self.gibbs_sample_ps()

    def get_labels(self, last_n_samples=None, prob=False, return_dict=True):
        """
        Get the core-periphery labels of each node. Lower indexed blocks are
        more core, e.g. in the hub-and-spoke model 0 is core, 1 is periphery
        and in the layered model 0 is the innermost layer

        Parameters
        ----------
        last_n_samples: int
            Infer labels over the last_n_samples in the Gibbs sample chain.
            Defaults to inferring over the entire chain
        prob: bool
            If True, return the probability a node belongs to each block
        return_dict: bool
            If True, return dictionary mapping of node to label. If False,
            return list of labels, useful if passing to MDL. If False and
            prob is True, returns n_nodes x n_blocks array with probabilities

        Returns
        -------
        If prob is False, returns a dictionary where node labels are keys and
        inferred block assignments are values. If prob is True, keys are ordered
        arrays of length n_blocks with the inferred probability of belonging
        to each block
        """
        if last_n_samples is None:
            n_samples = self.n_gibbs
        else:
            n_samples = last_n_samples

        if prob:
            inf_labels = [np.bincount(self.node_labels[-n_samples:,i],
                                      minlength=self.n_blocks) / n_samples
                          for i in self.nodes]
        else:
            inf_labels = [np.argmax(np.bincount(self.node_labels[-n_samples:,i]))
                          for i in self.nodes]

        if return_dict:
            node2label = {self.index2node[node_i]:l for node_i,l in enumerate(inf_labels)}
            return node2label
        else:
            return np.array(inf_labels)

    def get_coreness(self, last_n_samples=None, return_dict=True):
        """
        Get the coreness of each node, a continuous measure of a node's position
        in the core-periphery structure. The score varies between 0 and 1, where
        0 indicates nodes that are least core (i.e. positioned in the periphery)
        and 1 indicates nodes that are most core (i.e. positioned in the
        innermost layer). Note this is opposite of how the labels are
        interpretted: a core node will have a label of 0 and coreness of 1

        Parameters
        ----------
        last_n_samples: int
            Infer coreness over the last_n_samples in the Gibbs sample chain.
            Defaults to inferring over the entire chain
        return_dict: bool
            If True, return dictionary mapping of node to coreness. If False,
            return list of corenesses

        Returns
        -------
        If return_dict is True, returns a dictionary where node labels are keys
        and corenesses are values. Else returns an array the length of the
        number of nodes where entries are corenesses
        """
        if last_n_samples is None:
            n_samples = self.n_gibbs
        else:
            n_samples = last_n_samples

        avg_blocks = np.mean(self.node_labels[-n_samples:], axis=0)
        max_block = self.n_blocks - 1
        normed_cs = 1 - (avg_blocks / max_block)

        if return_dict:
            node2coreness = {self.index2node[node_i]:c for node_i,c in enumerate(normed_cs)}
            return node2coreness
        else:
            return normed_cs

    def clean_network(self, G):
        """
        Reindex the node labels as integers (maintaining alphabetic ordering)
        and removes self loops from the network. Returns a copy of the input.

        Parameters
        ----------
            G: NetworkX graph
        """
        # Relabel network to have integer labels
        H = nx.convert_node_labels_to_integers(G, ordering='sorted')
        self.nodes = list()
        self.node2index = dict()
        self.index2node = dict()
        for index,node in enumerate(sorted(G.nodes())):
            self.nodes.append(index)
            self.node2index[node] = index
            self.index2node[index] = node
            # Remove self loops
            if H.has_edge(index, index):
                H.remove_edge(index, index)

        return H

    def initialize_inference(self, G):
        """
        Initialize the inference of the core-periphery structure
           1. Randomly initialize node labels
           2. Calculate block statistics from node labels
           3. Reorder the blocks according to within layer densities
           4. Initialize ps having ordered the densities
           5. Calculate the log posterior using the ordered block stats
           6. Set the number of MCMC steps to take if not set by user

        Parameters
        ----------
        G: NetworkX graph
            Graph for which to infer a core-periphery structure. Assumes that
            the nodes are labeled according to 0-indexed integers
        """
        self.N_nodes = len(self.nodes)
        self.M_edges = len(G.edges())
        # Initialize the node labels as a random draw from the layers
        self.node_labels = np.zeros((self.n_gibbs, self.N_nodes), dtype=int)
        self.node_labels[0] = list(np.random.choice(self.blocks, size=self.N_nodes))
        # Get the ordered block statistics
        self.update_block_stats(G)
        # Reject initialization with empty blocks
        if np.sum(self.block_ns[0] == 0) > 0:
            return False
        # Initialize the latent ps now that you have ms and Ms
        self.gibbs_sample_ps()
        # Get the initial log posterior now that you have ns, ms, Ms, and ps
        self.log_labels_given_ps[0] = self.get_log_labels_given_ps()
        self.log_posteriors[0] = self.get_log_posterior()
        # Set number of MCMC steps if not set by user
        if self.n_mcmc is None:
            self.n_mcmc = N * len(self.n_blocks)

    def update_block_stats(self, G):
        """
        Updates blocks during the inference of core-peirphery structure
        according to the node labels at step T. Assumes that the node labels at
        time step T have already been initialized
        """
        # Order the blocks according to the within block densities
        if (self.mcmc_init == 'random') or (self.mcmc_init == 'previous' and self.T == 0):
            ordered_blocks = nh.get_ordered_block_stats(G, self.node_labels[self.T], self.n_blocks)
            self.node_labels[self.T] = ordered_blocks[0]
            self.block_ns[self.T] = ordered_blocks[1]
            self.block_ms[self.T] = ordered_blocks[2]
            self.block_Ms[self.T] = ordered_blocks[3]
        else:
            self.node_labels[self.T] = self.node_labels[self.T-1]
            self.block_ns[self.T] = self.block_ns[self.T-1]
            self.block_ms[self.T] = self.block_ms[self.T-1]
            self.block_Ms[self.T] = self.block_Ms[self.T-1]

    def get_log_posterior(self):
        """
        This needs to be defined specifically for each core-periphery class
        because each model has its own priors and, therefore, posterior
        """
        return

    def get_log_labels_given_ps(self):
        """
        This needs to be defined specifically for each core-periphery class
        because each model has its own priors
        """
        return

    def gibbs_sample_ps(self):
        """
        This needs to be defined specifically for each core-periphery class
        because each model has its own priors on the ps
        """
        return

    def gibbs_sample_labels(self, G):
        """
        Runs the Gibbs sampling step for the labels
        """
        # Redraw node labels and calculate statistics accordingly
        if self.mcmc_init == 'random':
            self.node_labels[self.T] = list(np.random.choice(self.blocks, size=self.N_nodes))
        elif self.mcmc_init == 'previous':
            self.node_labels[self.T] = self.node_labels[self.T-1]
        self.update_block_stats(G)
        # Calculate log posterior for evaluating the MH criterion
        self.log_posteriors[self.T] = self.get_log_posterior()
        self.log_labels_given_ps[self.T] = self.get_log_labels_given_ps()
        # Run MCMC on the node labels
        self.mcmc_sample_labels(G)

    def mcmc_sample_labels(self, G):
        """
        Runs an MCMC chain on the node labels
        """
        total_accept = 0
        nodes_for_mcmc = np.random.choice(self.nodes, size=self.n_mcmc, replace=True)
        for node_n,i in enumerate(nodes_for_mcmc):
            # Save state of blocks, in case MCMC step needs to be reverted
            saved = self.save_block_stats()

            # Propose move for MCMC and update the block statistics accordingly
            i_label = self.node_labels[self.T,i]
            if self.moves == 'neighbor':
                proposal = self.propose_neigh_move(G, i)
            elif self.moves == 'random':
                proposal = self.propose_random_move()
            self.update_blocks_proposed(G, i, proposal)

            # Evaluate move according to Metropolis-Hastings criterion
            prev_log_labels_given_ps = saved['log_labels_given_ps']
            acc = self.evaluate_label_move(G, i, i_label, proposal,
                                           prev_log_labels_given_ps)

            # Revert blocks if move is rejected
            total_accept += acc
            if acc == 0:
                self.revert_block_stats(saved)

        # Save the MCMC acceptance rate
        self.accepts[self.T-1] = total_accept / self.n_mcmc

    def update_blocks_proposed(self, G, i, proposal):
        """
        This needs to be defined specifically for each core-periphery class
        because each model needs to efficiently update different attriburtes
        """
        return

    def evaluate_label_move(self, G, i, i_label, proposal, prev_log_labels_given_ps):
        """
        Evaluates a proposed move in the MCMC sampling according to the
        Metropolis-Hastings criterion. Assumes block statistics have already
        been updated for time T to reflect the proposal

        Parameters
        ----------
        i: int
            Node for whcih we are proposing a new label
        i_label: int
            The label of node i before the moving to the proposed label
        proposal: int
            The proposed label for node i
        prev_log_labels_given_ps: float
            log P(\theta \mid p, A) from before the proposed update

        Returns
        -------
        acc: int
            Returns 1 if proposed move was accepted, returns 0 otherwise
        """
        # Get forward transition probabilities
        if self.moves == 'neighbor':
            log_update = self.get_transition_prob_neigh(G, i, proposal)
            log_revert = self.get_transition_prob_neigh(G, i, i_label)
        elif self.moves == 'random':
            log_update = self.get_transition_prob_random()
            log_revert = self.get_transition_prob_random()
        # Run proposed move through Metropolis-Hastings criterion
        # Reject moves that result in empty blocks
        if np.sum(self.block_ns[self.T] == 0) > 0:
            acc = 0
        else:
            acc = infh.check_MH_criterion(prev_log_labels_given_ps,
                                          self.log_labels_given_ps[self.T],
                                          log_update, log_revert)
        return acc

    def propose_random_move(self):
        """
        Returns a proposed MCMC move by uniformly drawing at random from all
        blocks
        """
        return np.random.choice(self.blocks)

    def get_transition_prob_random(self):
        """
        Returns the probability of transitioning from one parameter state to
        another when choosing a label at random for a random node
        """
        return np.log(1 / (self.N_nodes * self.n_blocks))

    def propose_neigh_move(self, G, i, T, blocks):
        """
        Returns a proposed label according to the labels of a node's neighbors
        """
        # Get a neighbor for making a move proposal
        j = np.random.choice(list(G[i]))
        j_block = self.node_labels[self.T,j]
        m_j_block = np.sum(self.block_ms[self.T,j_block])
        # Get the probability of each block connecting to j's label
        p_adj_blocks = np.zeros(self.n_blocks)
        for s in range(self.n_blocks):
            p_adj_blocks[s] = (self.block_ms[T,j_block,s] + self.eps)\
                              / (m_j_block + self.eps * self.n_blocks)
        # Propose move based on the probability of connecting to j's label
        proposal = np.random.choice(self.blocks, p=p_adj_blocks)

        return proposal

    def get_transition_prob_neigh(self, G, i, proposal):
        """
        Returns the probability of transitioning from one parameter state to
        another when proposing a move using the neighbor information
        $$
            \sum_s \sum_j
                \frac{
                    A_{ij} \delta_{\theta_{j,s}}
                }{
                    k_i
                }
                \fracc{
                    m{sr} + \eps
                }{
                    m_s + \eps B
                }
        $$
        where $A$ is the adjacency matrix, $\theta$ is the vector of node
        labels, $k_i$ is the degree of node $i$, $m_{sr}$ is the number of
        edges between blocks $r$ and $s$, $m_s$ is the totaly number of
        edges connected to block $s$, and $\eps$ is the parameter above.

        Parameters
        ----------
        i: int
            Node for which we are proposing a new label
        proposal: int
            Label that we are proposing for i
        """
        # The the local probabilities of node i of connecting to each block
        neigh_blocks2freq = Counter([self.node_labels[self.T,j] for j in G[i]])
        local_block_ps = [neigh_blocks2freq[b] if b in neigh_blocks2freq else 0
                          for b in self.blocks]
        local_block_ps = np.array(local_block_ps)/sum(neigh_blocks2freq.values())

        # Sum probability over all groups
        r = proposal
        p_transition = 0
        for s in self.blocks:
            p_random = (self.block_ms[self.T,s,r] + self.eps)\
                       /(np.sum(self.block_ms[self.T,s]) + self.eps * (self.n_blocks))
            p_transition += local_block_ps[s] * p_random

        return np.log(p_transition)

    def save_block_stats(self):
        """
        Wraps the state of the blocks at iteration T in the Gibbs sampling so
        that they can be saved
        """
        saved = dict()
        saved['node_labels'] = self.node_labels[self.T].copy()
        saved['block_ns'] = self.block_ns[self.T].copy()
        saved['block_ms'] = self.block_ms[self.T].copy()
        saved['block_Ms'] = self.block_Ms[self.T].copy()
        saved['log_posteriors'] = self.log_posteriors[self.T].copy()
        saved['log_labels_given_ps'] = self.log_labels_given_ps[self.T].copy()

        return saved

    def revert_block_stats(self, saved):
        """
        Reverts the state of blocks at iteration T in the Gibbs sampling back to
        the saved state
        """
        self.node_labels[self.T] = saved['node_labels']
        self.block_ns[self.T] = saved['block_ns']
        self.block_ms[self.T] = saved['block_ms']
        self.block_Ms[self.T] = saved['block_Ms']
        self.log_posteriors[self.T] = saved['log_posteriors']
        self.log_labels_given_ps[self.T] = saved['log_labels_given_ps']

# ------------------------------------------------------------------------------
# ------------------------ Layered core-periphery model ------------------------
# ------------------------------------------------------------------------------
class LayeredCorePeriphery(CorePeriphery):
    def __init__(self, n_layers, n_gibbs=200, n_mcmc=None, eps=1,
                 mcmc_init="random", moves="random", seed=None):
        """
        Parameters
        ----------
        n_layers: int
            The number of layers to infer for the core-periphery
        n_gibbs: int
            The number of steps to use in the Gibbs sampling between the ps and
            the node labels
        n_mcmc: int
            The number of steps to use in the MCMC sampling of the node labels
            within the Gibbs sampling. Defaults to n_nodes * n_blocks
        eps: float
            Parameter for neighbor-based MCMC proposals that is relatively
            arbitrary. As eps goes to infinity, the proposed moves become
            equivalent to choosing uniformaly at random from all blocks. Does
            not matter if moves="random". See `moves` parameter below.
        mcmc_init: str
            If "previous", then initialize each MCMC chain from the final
            state of the previous chain. If "random", initialize each chain
            randomly. The first initialization is always random.
        moves: str
            If "random", label proposals are drawn uniformly at random among all
            blocks during MCMC. If "neighbor", labels are proposed according to
            the neighbor labels. For details: Peixoto, T. P. (2014). Efficient
            Monte Carlo and greedy heuristic for the inference of stochastic
            block models. *Physical Review E*.
        seed: int
            Seed passed to np.random.seed()
        """
        # Initialize inference parameters and data structures
        super().__init__(n_layers, n_gibbs, n_mcmc, eps, mcmc_init, moves, seed)
        self.n_layers = n_layers
        # Initialize data structures specific for layered inference
        self.layer_ms = np.zeros((self.n_gibbs, self.n_layers), dtype=np.int64)
        self.layer_Ms = np.zeros((self.n_gibbs, self.n_layers), dtype=np.int64)
        self.layer_ps = np.zeros((self.n_gibbs, self.n_layers))

    def get_log_posterior(self):
        """
        Calculates the log posterior of the layered core-periphery model
        """
        # When T == 0, we're initializing everything on the same step
        if self.T == 0:
            return infh.get_log_posterior_layered(self.N_nodes,
                                                  self.layer_ms[self.T],
                                                  self.layer_Ms[self.T],
                                                  self.block_ns[self.T],
                                                  self.layer_ps[self.T])
        # Otherwise, we're updating the labels first (T), using the sample of
        # the ps from the previous iteration (T-1)
        else:
            return infh.get_log_posterior_layered(self.N_nodes,
                                                  self.layer_ms[self.T],
                                                  self.layer_Ms[self.T],
                                                  self.block_ns[self.T],
                                                  self.layer_ps[self.T-1])

    def get_log_labels_given_ps(self):
        """
        Calculates log P(\theta \mid p, A) for the layered model
        """
        # When T == 0, we're initializing everything on the same step
        if self.T == 0:
            return infh.log_labels_given_ps_layered(self.N_nodes,
                                                    self.block_ns[self.T],
                                                    self.layer_ms[self.T],
                                                    self.layer_Ms[self.T],
                                                    self.layer_ps[self.T])
        # Otherwise, we're updating the labels first (T), using the sample of
        # the ps from the previous iteration (T-1)
        else:
            return infh.log_labels_given_ps_layered(self.N_nodes,
                                                    self.block_ns[self.T],
                                                    self.layer_ms[self.T],
                                                    self.layer_Ms[self.T],
                                                    self.layer_ps[self.T-1])

    def update_block_stats(self, G):
        """
        Updates blocks during the inference of core-peirphery structure
        according to the node labels at step T. Assumes that the node labels at
        time step T have already been initialized
        """
        super().update_block_stats(G)
        # Get layer statistics now that we have the ordered block statistics
        if (self.mcmc_init == 'random') or (self.mcmc_init == 'previous' and self.T == 0):
            ordered_layers = nh.get_layered_stats(self.block_ns[self.T], self.block_ms[self.T])
            self.layer_ms[self.T] = ordered_layers[0]
            self.layer_Ms[self.T] = ordered_layers[1]
        else:
            self.layer_ms[self.T] = self.layer_ms[self.T-1]
            self.layer_Ms[self.T] = self.layer_Ms[self.T-1]

    def gibbs_sample_ps(self):
        """
        Gibbs samples the densities of layers given the edge counts of the
        layers and updates them in the object accordingly
        """
        for s in self.blocks:
            n_s = self.block_ns[self.T,s]
            m_s = self.layer_ms[self.T,s]
            M_s = self.layer_Ms[self.T,s]

            # Sampling innermost layer
            if s == 0:
            	self.layer_ps[self.T,s] = infh.sample_trunc_beta(m_s+1,
                                                                 M_s-m_s+1,
                                                                 self.layer_ps[self.T-1,1],
                                                                 1)
            # Sampling outermost layer
            elif s == self.n_layers - 1:
                self.layer_ps[self.T,s] = infh.sample_trunc_beta(m_s+1,
                                                                 M_s-m_s+1,
                                                                 0,
                                                                 self.layer_ps[self.T,s-1])
            # Sampling all other layers
            else:
                self.layer_ps[self.T,s] = infh.sample_trunc_beta(m_s+1,
                                                                 M_s-m_s+1,
                                                                 self.layer_ps[self.T-1,s+1],
                                                                 self.layer_ps[self.T,s-1])

    def update_blocks_proposed(self, G, i, proposal):
        """
        Calculates the statistics of the group sizes and edge counts for a given
        proposed move. Given the current data structures, this can be done more
        quickly here than starting from scratch
            1. Updates block_ns
            2. Updates block_Ms using updated block_ns
            3. Updates block_ms and layer_ms using i's neighbors
            4. Updates the log posterior using the updated statistics

        Parameters
        ----------
        i: int
            The node whose label we are proposing a move for
        proposal: int
            The proposed label for node i
        """
        # Update node's label itself
        i_label = self.node_labels[self.T,i]
        self.node_labels[self.T,i] = proposal
        # Get the updated block sizes
        self.block_ns[self.T,i_label] -= 1
        self.block_ns[self.T,proposal] += 1
        # Get the updated number of potential edges between blocks
        self.layer_Ms[self.T] = np.zeros(self.n_layers)
        for r in self.blocks:
            for s in range(r+1):
                self.layer_Ms[self.T,r] += nh.get_max_edges(r, s, self.block_ns[self.T])
        # Get the updated number of existing edges between blocks
        for j in G[i]:
            # Identify layers that the proposal affects
            j_label = self.node_labels[self.T,j]
            max_before_label = max(i_label, j_label)
            max_after_label = max(proposal, j_label)
            # Update the layer edge counts
            self.layer_ms[self.T,max_before_label] -= 1
            self.layer_ms[self.T,max_after_label] += 1
            # Update the block edge counts
            self.block_ms[self.T,i_label,j_label] -= 1
            if i_label != j_label:
                self.block_ms[self.T,j_label,i_label] -= 1
            self.block_ms[self.T,proposal,j_label] += 1
            if proposal != j_label:
                self.block_ms[self.T,j_label,proposal] += 1
        # Update the log posterior
        self.log_posteriors[self.T] = self.get_log_posterior()
        self.log_labels_given_ps[self.T] = self.get_log_labels_given_ps()

    def save_block_stats(self):
        """
        Wraps the state of the blocks at iteration T in the Gibbs sampling so
        that they can be saved
        """
        saved = super().save_block_stats()
        saved['layer_ms'] = self.layer_ms[self.T].copy()
        saved['layer_Ms'] = self.layer_Ms[self.T].copy()
        saved['layer_ps'] = self.layer_ps[self.T].copy()

        return saved

    def revert_block_stats(self, saved):
        """
        Reverts the state of blocks at iteration T in the Gibbs sampling back to
        the saved state
        """
        super().revert_block_stats(saved)
        self.layer_ms[self.T] = saved['layer_ms']
        self.layer_Ms[self.T] = saved['layer_Ms']
        self.layer_ps[self.T] = saved['layer_ps']

# ------------------------------------------------------------------------------
# --------------------- Hub-and-spoke core-periphery model ---------------------
# ------------------------------------------------------------------------------
class HubSpokeCorePeriphery(CorePeriphery):
    def __init__(self, n_gibbs=200, n_mcmc=None, eps=1, mcmc_init="random",
                 moves="random", seed=None):
        """
        Parameters
        ----------
        n_gibbs: int
            The number of steps to use in the Gibbs sampling between the ps and
            the node labels
        n_mcmc: int
            The number of steps to use in the MCMC sampling of the node labels
            within the Gibbs sampling. Defaults to n_nodes * n_blocks
        eps: float
            Parameter for neighbor-based MCMC proposals that is relatively
            arbitrary. As eps goes to infinity, the proposed moves become
            equivalent to choosing uniformaly at random from all blocks. Does
            not matter if moves="random". See `moves` parameter below.
        mcmc_init: str
            If "previous", then initialize each MCMC chain from the final
            state of the previous chain. If "random", initialize each chain
            randomly. The first initialization is always random.
        moves: str
            If "random", label proposals are drawn uniformly at random among all
            blocks during MCMC. If "neighbor", labels are proposed according to
            the neighbor labels. For details: Peixoto, T. P. (2014). Efficient
            Monte Carlo and greedy heuristic for the inference of stochastic
            block models. *Physical Review E*.
        seed: int
            Seed passed to np.random.seed()
        """
        # Initialize inference parameters and data structures
        super().__init__(2, n_gibbs, n_mcmc, eps, mcmc_init, moves, seed)
        # Initialize data structures specific for hub+spoke inference
        self.block_ps = np.zeros((self.n_gibbs, self.n_blocks, self.n_blocks))

    def get_log_posterior(self):
        """
        Calculates the log posterior of the hub-and-spoke core-periphery model
        """
        # When T == 0, we're initializing everything on the same step
        if self.T == 0:
            return infh.get_log_posterior_hubspoke(self.N_nodes,
                                                   self.block_ms[self.T],
                                                   self.block_Ms[self.T],
                                                   self.block_ns[self.T],
                                                   self.block_ps[self.T])
        # Otherwise, we're updating the labels first (T), using the sample of
        # the ps from the previous iteration (T-1)
        else:
            return infh.get_log_posterior_hubspoke(self.N_nodes,
                                                   self.block_ms[self.T],
                                                   self.block_Ms[self.T],
                                                   self.block_ns[self.T],
                                                   self.block_ps[self.T-1])

    def get_log_labels_given_ps(self):
        """
        Calculates log P(\theta \mid p, A) for the hub-and-spoke model
        """
        # When T == 0, we're initializing everything on the same step
        if self.T == 0:
            return infh.log_labels_given_ps_hubspoke(self.N_nodes,
                                                     self.block_ns[self.T],
                                                     self.block_ms[self.T],
                                                     self.block_Ms[self.T],
                                                     self.block_ps[self.T])
        # Otherwise, we're updating the labels first (T), using the sample of
        # the ps from the previous iteration (T-1)
        else:
            return infh.log_labels_given_ps_hubspoke(self.N_nodes,
                                                     self.block_ns[self.T],
                                                     self.block_ms[self.T],
                                                     self.block_Ms[self.T],
                                                     self.block_ps[self.T-1])

    def gibbs_sample_ps(self):
        """
        Gibbs samples the densities of blocks given the edge counts of the
        blocks and updates them in the object accordingly
        """
        for r in range(2):
            for s in range(r+1):
                m_s = self.block_ms[self.T,s,r]
                M_s = self.block_Ms[self.T,s,r]

                # Sample the core
                if r == 0 and s == 0:
                    self.block_ps[self.T,r,s] = infh.sample_trunc_beta(m_s+1,
                                                                       M_s-m_s+1,
                                                                       self.block_ps[self.T-1,0,1],
                                                                       1)
                # Sample the periphery
                elif r == 1 and s == 1:
                    self.block_ps[self.T,r,s] = infh.sample_trunc_beta(m_s+1,
                                                                       M_s-m_s+1,
                                                                       0,
                                                                       self.block_ps[self.T,0,1])
                # Sample core-to-periphery
                else:
                    p = infh.sample_trunc_beta(m_s+1,
                                               M_s-m_s+1,
                                               self.block_ps[self.T-1,1,1],
                                               self.block_ps[self.T,0,0])
                    self.block_ps[self.T,r,s] = p
                    self.block_ps[self.T,s,r] = p

    def update_blocks_proposed(self, G, i, proposal):
        """
        Calculates the statistics of the group sizes and edge counts for a given
        proposed move. This can be done faster here than starting from scratch
            1. Updates block_ns
            2. Updates block_Ms using updated block_ns
            3. Updates block_ms using i's neighbors
            4. Updates the log posterior using the updated statistics

        Parameters
        ----------
        i: int
            The node whose label we are proposing a move for
        proposal: int
            The proposed label for node i
        """
        # Update node's label itself
        i_label = self.node_labels[self.T,i]
        self.node_labels[self.T,i] = proposal
        # Get the updated block sizes
        self.block_ns[self.T,i_label] -= 1
        self.block_ns[self.T,proposal] += 1
        # Update the potential block edge counts
        self.block_Ms[self.T] = np.zeros((self.n_blocks,self.n_blocks))
        for r in self.blocks:
            for s in range(r+1):
                M_rs = nh.get_max_edges(r, s, self.block_ns[self.T])
                self.block_Ms[self.T,r,s] = M_rs
                if r != s:
                    self.block_Ms[self.T,s,r] = M_rs
        # Get the updated number of existing edges between blocks
        for j in G[i]:
            # Identify blocks that the proposal affects
            j_label = self.node_labels[self.T,j]
            # Update the block edge counts
            self.block_ms[self.T,i_label,j_label] -= 1
            if i_label != j_label:
                self.block_ms[self.T,j_label,i_label] -= 1
            self.block_ms[self.T,j_label,proposal] += 1
            if j_label != proposal:
                self.block_ms[self.T,proposal,j_label] += 1
        # Update the log posterior
        self.log_posteriors[self.T] = self.get_log_posterior()
        self.log_labels_given_ps[self.T] = self.get_log_labels_given_ps()
