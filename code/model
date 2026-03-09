import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
from gcn_model import GCN

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def forward(self, input_ids=None,labels=None): 
        # print('input_ids:', input_ids.shape)
        # print('labels:', labels.shape)
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob


class Modelwithcfgdfg(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Modelwithcfgdfg, self).__init__()
        self.encoder = encoder
        self.cfg_encoder = GCN(in_channels=300, hidden_channels=1024, out_channels=768, pooling_type=args.pooling_type)  
        self.dfg_encoder = GCN(in_channels=300, hidden_channels=1024, out_channels=768, pooling_type=args.pooling_type)  
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)
        #self.classifier = nn.Linear(768, 1)
        if self.args.only_cfg:
            self.classifier = nn.Linear(768 * 2, 1)  # 假设隐藏维度为 768 * 2，二分类
            print("classifier using 768*2")
        elif self.args.only_dfg:
            self.classifier = nn.Linear(768 * 2, 1)  # 假设隐藏维度为 768 * 2，二分类
            print("classifier using 768*2")
        else:
            self.classifier = nn.Linear(768 * 3, 1)  # 假设隐藏维度为 768 * 3，二分类
            print("classifier using 768*3")

        
    def forward(self, input_ids=None,labels=None,cfg=None,dfg=None): 
        #outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        # print('input_ids:', input_ids.shape)
        # print('label:', labels.shape)
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1), output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]  
        #print("last_hidden_state:", last_hidden_state.shape)
        # 取 [CLS] token 的嵌入，通常在位置 0  
        cls_embedding = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]  
        #print("cls_embedding:", cls_embedding.shape)  

        # 用 cls_embedding 代替原来的 outputs  
        outputs = cls_embedding  # Now outputs has shape [128, hidden_size]
        #print("outputs:", outputs.shape)

        if not self.args.only_dfg:
            cfg_embedding = self.cfg_encoder(cfg.x, cfg.edge_index, cfg.batch)
            #print("cfg_embedding:", cfg_embedding.shape)
            #outputs += cfg_embedding
            outputs = torch.cat((outputs, cfg_embedding), dim=-1) 
        if not self.args.only_cfg:
            dfg_embedding = self.dfg_encoder(dfg.x, dfg.edge_index, dfg.batch)
            #print("dfg_embedding:", dfg_embedding.shape)
            #outputs += dfg_embedding
            outputs = torch.cat((outputs, dfg_embedding), dim=-1) 
        
        # Apply dropout
        outputs = self.dropout(outputs)

        logits=self.classifier(outputs)  
        prob=torch.sigmoid(logits)
        #print("prob:", prob.shape)
        if labels is not None:
            labels=labels.float()
            #print("labels:", labels.shape)
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            #print("loss:", loss.shape)
            loss=-loss.mean()
            #print("loss:", loss.shape)
            return loss,prob
        else:
            return prob
      
        
 
