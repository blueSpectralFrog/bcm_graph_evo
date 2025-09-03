import jax
import jax.numpy as jnp
import optax
import equinox
import networkx
import graph_utils as gu
import encoder
import pandas as pd

dataset = pd.read_csv(r'C:\Users\ndnde\Documents\Projects\ML\datasets\OCEAN attributes\OCEAN-synthetic_topics.csv')

# parameters
N = len(dataset) # number of nodes 
T_steps = 500
K = 15
lr = 1e-4

# random keys
rootKey = jax.random.key(0)
rootKey, subkey = jax.random.split(rootKey)
rootKey, graphkey = jax.random.split(subkey)

# init node state
OCEAN_raw = dataset.iloc[:,2:7]
OCEAN = jnp.array(OCEAN_raw.div(OCEAN_raw.sum(axis=1), axis=0))
topics = jnp.array(dataset.iloc[:,7:]) # unsure whether to normalize
opinions = jax.random.uniform(rootKey, shape=[N,N]) # opinions of every other node
opinions = jnp.fill_diagonal(opinions, 1, inplace=False) # every node likes itself
# probably don't need opinion of EVERY other node, only the ones that it's close too or touches

# Create latent encoding h of OCEAN into latent space of "topics"
feat_encoder = encoder.Encoder(jnp.size(jnp.concatenate([OCEAN, opinions],-1).T,0), 
                               64,
                               topics.shape[1])
vmap_encoder = jax.vmap(feat_encoder, in_axes=(0,))
h = vmap_encoder(jnp.concatenate([OCEAN, opinions],-1)) # initiate the weights. will probably move this into training loop

init_graph = gu.generate_er_graph_jax(N, p=0.5, seed=None) # N*N adjacency matrix
# gu.visualize_graph(init_graph) # takes a while for all the nodes. Use init_graph[:n,:n].

# Initiate MLP with input size size_OCEAN + size_h (from OCEAN)
in_size = 5 + topics.shape[1]
out_size = 3
width_size = 32
depth = 2
activation = jax.nn.relu
phiMLP = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation)

# Initiate epsilon and nu MLPs, responsible for interpreting message by receiver node
in_size = 2 + out_size
out_size = 3
width_size = 32
depth = 2
activation = jax.nn.relu
epsMLP = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation)
nuMLP = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation)


@jax.jit
def step():
    """
    1. node r recieves tweets from surrounding nodes, phi(OCEAN_s, ENC(OCEAN_s, O_rs)
    2. node r "sees" tweet and interprets it 
        2.1 msr' = O_rs*msr
        2.2 epsMLP(O, A, msr'), nuMLP(C, N, msr')
        2.3 opinion update 
            - O_rs' = relu(O_rs + nuMLP(O_rs) + noise(?) if O_rs < epsMLP
            - O_rs' = relu(O_rs + noise) otherwise
        
    return O_rs'
    """
    pass


def loss_fn(K, OCEAN, opinions, init_graph, topics, subkey, eps=1e-8):
    h_embed = phiMLP(OCEAN, opinions)
    losses = []    

    for t in range(K):
        opinions_new, aux = step(params, state, graph=None, rng=rng)
        h_embed_new = phiMLP(OCEAN, opinions_new)
        loss = jnp.sum(1-jnp.abs(h_embed-h_embed_new)/(h_embed+h_embed_new+eps))/h_embed.shape[0]
    return jnp.sum(loss)

opt = optax.adam(lr)
state0 = [OCEAN, opinions]
for step in T_steps:
    rng, sub = jax.random.split(rng)
    loss, grads = equinox.filter_value_and_grad(loss_fn)(K, OCEAN, opinions, init_graph, topics, subkey)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = equinox.apply_updates(params, updates)
"""
@jax.jit
def message_pass(features, graph):
    features -> encode 

    take node batch -> talk to neighbours -> aggregate 
    take embedded node batch -> talk to "neighbors" -> aggregate -> decode?

    take aggregated node features + decoded embedded node features -> update features

    return updated_features

def loss(updated_features,features):
    loss = RMSE(updated_features,features) ...need better loss
    return loss

for 1:T_steps:
    for 1:Batches:
        updated_features = message_pass(fetures)
        loss = loss(updated_features,features)
"""