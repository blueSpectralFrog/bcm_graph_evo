The purpose of this project is to learn the behavior of nodes based on their 
"Big 5" character traits when put into contact with other nodes. The setting 
is meant to imitate a social media like X, where nodes will develop relationships
with other nodes based off of common interests, but will only share connections 
(edges in the graph) with nodes who align with them based on their OCEAN 
characteristics. 

The basic idea is something like this: 
1. Create a random graph of nodes with OCEAN attributes as node features
2. Map OCEAN into "topics" using a Bounded Confidence Model (BCM) which 
apparently is a common way to model social interaction. Topics are pre-labeled using 
OCEAN-synthetic dataset from huggingface. 
3. Use these topics as a metric for how well nodes "get along" with each other
and how likely they are to develop an edge.
4. Embed the OCEAN attributes into a higher dimension using an decoder structure
to project their topics onto various subjects. This allows clustering of nodes 
in the latent space even if in the "real world" (graph space) they do not share an
edge.
5. Apply message passing to the graph using some kind of trait-conditioning as a filter
for what messages are accepted and which rejected by a node. 