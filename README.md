# Bayesian Core-Periphery Stochastic Block Models

Core-periphery structure is one of the most ubiquitous mesoscale patterns in networks. This code implements two Bayesian stochastic block models in Python for modeling hub-and-spoke and layered core-periphery structures. It can be used for probabilistic inference of core-periphery structure and model selection between the two core-periphery characterizations.

If you use this code, please cite the following:

> Gallagher, Ryan J., Young, Jean-Gabriel, and Foucault Welles, Brooke. (2020). "[A Clarified Typology of Core-Periphery Structure in Networks](https://arxiv.org/abs/2005.10191)." arXiv preprint.

## Running Core-Periphery Models

The models can be run in Python with any NetworkX graph. To infer a hub-and-spoke structure:

```python
import networkx as nx
from core_periphery_sbm import core_periphery as cp

# Load Les Mis graph from NetworkX
G = nx.les_miserables_graph()

# Initialize hub-and-spoke model and infer structure
hubspoke = cp.HubSpokeCorePeriphery(n_gibbs=100, n_mcmc=10*len(G))
hubspoke.infer(G)
```

Similarly, we can infer a layered structure with `n_layers`:

```python
layered = cp.LayeredCorePeriphery(n_layers=3, n_gibbs=100, n_mcmc=10*len(G))
layered.infer(G)
```

The models are inferred through an MCMC-within-Gibbs sampling procedure. There are `n_gibbs` samples that are taken, and during each Gibbs sample, an MCMC chain of `n_mcmc` steps is run on the labels of the network.


From the Gibbs samples, we can infer the core-periphery assignment of each node. Lower indices indicate *more* core positions. For example, in the hub-and-spoke model, 0 indicates the core and 1 indicates the periphery. In the layered model, 0 is the innermost layer.

```python
# Get core and periphery assignments from hub-and-spoke model
node2label_hs = hubspoke.get_labels(last_n_samples=50)

# Get layer assignments from the layered model
node2label_l = layered.get_labels(last_n_samples=50)
```

You can estimate the probability a node belongs to the core or periphery (hub-and-spoke model) or each layer (layered model). You can also retrieve just an array of labels or probabilities, where nodes are indexed according to their labels' alphabetic order in the network.

```python
#  Dictionary of node -> ordered array of probabilities
node2probs_hs = hubspoke.get_labels(last_n_samples=50, prob=True, return_dict=True)

# n_nodes x n_layers array of probabilities
inf_probs_l = layered.get_labels(last_n_samples=50, prob=True, return_dict=False)
```

## Core-Periphery Model Selection

Given core-periphery labels inferred from the hub-and-spoke and layered models, we want to determine which partition best describes the core-periphery structure of a network. We can do this by comparing the description length of the models. The *smaller* the description length, the *better* the model fit. We can approximate the description length for each core-periphery model using `n_samples`:

```python
from core_periphery_sbm import model_fit as mf

# Get description length of hub-and-spoke model
inf_labels_hs = hubspoke.get_labels(last_n_samples=50, prob=False, return_dict=False)
mdl_hubspoke = mf.mdl_hubspoke(G, inf_labels_hs, n_samples=100000)

# Get the description length of layered model
inf_labels_l = layered.get_labels(last_n_samples=50, prob=False, return_dict=False)
mdl_layered = mf.mdl_layered(G, inf_labels_l, n_layers=3, n_samples=100000)
```

## Further Reading

For more details about the hub-and-spoke and layered core-periphery block models and how they are inferred, please see:

> Gallagher, Ryan J., Young, Jean-Gabriel, and Foucault Welles, Brooke. (2020). "[A Clarified Typology of Core-Periphery Structure in Networks](https://arxiv.org/abs/2005.10191)." arXiv preprint.
