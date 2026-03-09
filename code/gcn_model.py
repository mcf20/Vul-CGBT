
import torch    
import torch.nn.functional as F    
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool        
from torch_geometric.data import Data, DataLoader, Batch    
from gensim.models import KeyedVectors    
import numpy as np  
  
# Load Google News pre-trained model    
model_path = '../data/GoogleNews-vectors-negative300.bin.gz'    
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)    
        
# Convert text to embeddings using Word2Vec    
def text_to_embedding(text, model): 
    if text is None:
        return torch.zeros(model.vector_size, dtype=torch.long)   
    words = text.split()    
    embeddings = [model[word] for word in words if word in model]    
    if embeddings:    
        return torch.tensor(sum(embeddings) / len(embeddings), dtype=torch.long)    
    else:    
        return torch.zeros(model.vector_size, dtype=torch.long)  
  
# Define GCN model    
class GCN(torch.nn.Module):    
    def __init__(self, in_channels, hidden_channels, out_channels, pooling_type="mean"):    
        super(GCN, self).__init__()  
        self.conv1 = GCNConv(in_channels, hidden_channels)    
        self.conv2 = GCNConv(hidden_channels, out_channels)    
        self.pooling_type = pooling_type
        
    def forward(self, x, edge_index, batch): 
        x = x.float()  
        edge_index = edge_index.long()    
        x = self.conv1(x, edge_index)    
        x = F.relu(x)    
        x = self.conv2(x, edge_index)   
        if self.pooling_type == "mean":
            x = global_mean_pool(x, batch)  # [num_graphs, out_channels]  
        elif self.pooling_type == "max":
            x = global_max_pool(x, batch)  # [num_graphs, out_channels] 
        elif self.pooling_type == "joint":
            x = global_add_pool(x, batch) * global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        #转成long类型
        #x = x.long()
        return x 
  
# Function to build a list of Data objects for CFG graphs  
def build_cfg_data_list(cfg_nodes_list, cfg_edges_list):    
    data_list = []    
    for nodes_text, edge_index in zip(cfg_nodes_list, cfg_edges_list):    
        # Ensure node features are float tensors  
        if nodes_text.ndim == 1:  
            nodes_text = nodes_text.unsqueeze(0)  
        data = Data(x=nodes_text, edge_index=edge_index)  
        data_list.append(data)    
  
    batch = Batch.from_data_list(data_list)    
    return batch  
  
# Function to build a list of Data objects for DFG graphs  
def build_dfg_data_list(dfg_nodes_list, dfg_edges_list):    
    data_list = []    
    for nodes_text, edge_index in zip(dfg_nodes_list, dfg_edges_list):    
        # Ensure node features are float tensors  
        if nodes_text.ndim == 1:  
            nodes_text = nodes_text.unsqueeze(0)  
        data = Data(x=nodes_text, edge_index=edge_index)  
        data_list.append(data)    
  
    batch = Batch.from_data_list(data_list)    
    return batch  
