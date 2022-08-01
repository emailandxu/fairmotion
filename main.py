#%%
import gzip
import pickle


#%%
with gzip.open(r"temp_motion_graph.gzip", "rb") as f:
    mg = pickle.load(f)

