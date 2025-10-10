import torch
import torch.nn as nn
class swigluffn(nn.Module):
    def __init__(self,d,dff):
        super().__init__()
        self.w1=nn.Linear(d,dff)
        self.w2=nn.Linear(d,dff)
        self.w3=nn.Linear(dff,d)
    def forward(self,x):
        return self.w3((self.w2(x))*(nn.functional.silu(self.w1(x))))
class MOE(nn.Module):
    def __init__(self,noe,contextlenght,cfg):
        super().__init__()
        self.contextlenght=contextlenght
        self.routing=nn.Linear(cfg["emb_dim"],noe)
        self.list_of_experts=nn.ModuleList([swigluffn(d=cfg["emb_dim"],dff=cfg["dff"]) for i in range(noe)])
    def forward(self,x):
        b,s,d=x.shape
        x=x.view(b*s,d)
        routing_vector=self.routing(x)
        priority,index=routing_vector.topk(k=2,dim=-1)
        priority=nn.functional.softmax(priority,dim=-1)
        y=[]
        for i in range(b*s):
                e1=(self.list_of_experts[index[i][0]](x[i]))*priority[i][0]
                e2=(self.list_of_experts[index[i][1]](x[i]))*priority[i][1]
                out=e1+e2
                y.append(out)
        return torch.stack(y,dim=0).view(b,s,d)
