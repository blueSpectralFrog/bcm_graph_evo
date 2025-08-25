import jax
import jax.numpy as jnp
import optax
import equinox
import networkx
import graph_utils as gu
import encoder

N = 100 # number of nodes 

# random keys
rootKey = jax.random.key(0)
rootKey, subkey = jax.random.split(rootKey)
rootKey, graphkey = jax.random.split(subkey)

# init node state
OCEAN = jax.random.uniform(rootKey, shape=[N,5])
opinions = jax.random.uniform(subkey, shape=[N,1])

# Create opinions about 10 topics. See README file for topics and OCEAN filters.
topics = jax.nn.softmax(jax.random.uniform(graphkey, shape=[10,1]), axis=0)

# Create latent encoding of OCEAN, opinions and topics
h = encoder.encoder(jnp.concat([OCEAN, opinions, topics],-1))

init_graph = gu.generate_er_graph_jax(N, p=0.5, seed=None) # N*N adjacency matrix

# eps = softplus(f_eps(z)); mu = sigmoid(f_mu(z)); gam = sigmoid(f_gam(z))

# E = initial_random_edges(N, avg_deg)

# Initiate MLP with input size 5 (from OCEAN)
in_size = 5
out_size = 3
width_size = 6
depth = 2
activation = jax.nn.relu
mlp = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation)


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