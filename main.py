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

# Create latent encoding h of OCEAN and opinion into latent space of "topics"
feat_encoder = encoder.Encoder(jnp.size(jnp.concatenate([OCEAN, opinions],-1).T,0), 
                               64,
                               topics.shape[1])
vmap_encoder = jax.vmap(feat_encoder, in_axes=(0,))
h0 = vmap_encoder(jnp.concatenate([OCEAN, opinions],-1)) # initiate the weights. will probably move this into training loop

init_graph, edges = gu.generate_er_graph_jax(N, p=0.004, seed=None) # N*N adjacency matrix
# gu.visualize_graph(init_graph) # takes a while for all the nodes. Use init_graph[:n,:n].

# Initiate MLP with input size size_OCEAN + size_h (from OCEAN)
in_size = OCEAN.shape[1] + topics.shape[1]
out_size = 3
width_size = 32
depth = 2
activation = jax.nn.relu
phiMLP = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation, key=subkey)

# Initiate epsilon and nu MLPs, responsible for interpreting message by receiver node
in_size = 2 + out_size
out_size = 1
width_size = 32
depth = 2
activation = jax.nn.relu
epsMLP = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation, key=subkey)
nuMLP = equinox.nn.MLP(in_size=in_size, out_size=out_size, width_size=width_size, depth=depth, activation=activation, key=subkey)
   
@equinox.filter_jit
def step(params, state):
    """
    1. node r recieves tweets from surrounding nodes, phi(OCEAN_s, ENC(OCEAN_s, O_rs)
    2. node r "sees" tweet and interprets it 
        2.1 msr' = O_rs*msr
        2.2 epsMLP(O, A, msr'), nuMLP(C, N, msr')
        2.3 opinion update 
            - O_rs' = relu(O_rs + nuMLP(O_rs) + noise(?) if O_rs < epsMLP
            - O_rs' = relu(O_rs + noise) otherwise
        2.4 inject 
        
    return O_rs'
    """
    OCEAN, opinions, edges, h = state
    phiMLP, epsMLP, nuMLP = params

    senders, receivers = jnp.array([s[1] for s in list(edges)]), jnp.array([r[0] for r in list(edges)])
    OCEAN_s, OCEAN_r = OCEAN[senders], OCEAN[receivers]
    opinion_rs = opinions[receivers, senders]
    h_embed_s, h_embed_r = h[senders], h[receivers]

    # 1. node r recieves tweets from surrounding nodes, phi(OCEAN_s, ENC(OCEAN_s, O_rs))
    vmap_phiMLP = jax.vmap(phiMLP)
    m_sr = vmap_phiMLP(jnp.concat((OCEAN_s, h_embed_s), axis=-1))
    
    # 2. node r "sees" tweet and interprets it 
    # 2.1
    m_sr_p = (opinion_rs * m_sr.T).T

    # 2.2 
    vmap_epsMLP = jax.vmap(epsMLP)
    vmap_nuMLP = jax.vmap(nuMLP)
    confidence = vmap_epsMLP(jnp.concat((OCEAN_r[:,[0]], OCEAN_r[:,[3]], m_sr_p), axis=-1)).squeeze()
    flexibility = vmap_nuMLP(jnp.concat((OCEAN_r[:,[1]], OCEAN_r[:,[4]], m_sr_p), axis=-1)).squeeze()
    # noise = (OCEAN[:,[2]]-0.5)*2 # ?
    # 2.3
    opinion_condition = opinion_rs < confidence
    if_true = jax.nn.relu(opinion_rs + flexibility * opinion_rs)# + noise)
    if_false = jax.nn.relu(opinion_rs)# + noise)
    opinion_rs_p = jnp.where(opinion_condition, if_true, if_false)
    # 2.4 Inject
    # inject opinion_rs_p back into N*N format
    opinions = opinions.at[receivers, senders].set(opinions[receivers, senders] + opinion_rs_p)

    return opinions

# @equinox.filter_jit
def loss_fn(K, params, state, topics, subkey, eps=1e-8):
    losses = []
    # subkey for remeshing later maybe
    
    for t in range(K):
        opinions_new = step(params, state)
        h_embed_new = vmap_encoder(jnp.concatenate([OCEAN, opinions_new],-1))
        state = OCEAN, opinions_new, list(edges), h_embed_new
        losses.append(jnp.sum(1-jnp.abs(topics-h_embed_new)/(topics+h_embed_new+eps))/topics.shape[0])
    return jnp.sum(jnp.array(losses))


# Initialize parameters and state
state0 = OCEAN, opinions, list(edges), h0
params = phiMLP, epsMLP, nuMLP
# Initialize optimizer
opt = optax.adam(lr)
opt_state = opt.init(equinox.filter(params, equinox.is_array))

for epoch in range(T_steps):
    rng, sub = jax.random.split(graphkey)
    loss, grads = equinox.filter_value_and_grad(loss_fn)(K, params, state0, topics, sub)
    updates, opt_state = opt.update(grads, opt_state)
    params = equinox.apply_updates(params, updates)
