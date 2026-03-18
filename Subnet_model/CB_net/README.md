`build_cb_net.py` builds a 500-neuron CB subnet with:

- `L0 -> L1 -> L2 -> L3` feedforward backbone
- weak feedback edges at reduced density/weight
- local clustered edges with exponential distance decay
- peduncle broadcast edges
- hypostome coordination edges

Outputs are written to [`outputs`](/Users/ASUS/Documents/Hydra/Subnet_model/CB_net/outputs):

- `cb_nodes.csv`
- `cb_edges.csv`
- `cb_summary.json`
- `cb_network_2d.png`
- `cb_network_3d.png`
