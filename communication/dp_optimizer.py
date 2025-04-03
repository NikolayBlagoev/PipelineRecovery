from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch import no_grad, cat, zeros_like, split, mean
from time import time
from torch.optim import Adam

class DP_optim(object):
    def __init__(self, lr, model, dp_group, device):
        super().__init__()
        self.dp_group = dp_group
        self.lr = lr
        self.iteration = 0
        self.optimizer = Adam(params=model.parameters(), lr=lr)
        self.net = model
        self.sizes = []
        self.device = device
        
        self.len_sizes = []
        for param in self.net.parameters():
            self.sizes.append(param.shape)
            self.len_sizes.append(len(param.view(-1)))
        
    def step(self):
        tmp = []
        
        with no_grad():
            
            for param in self.net.parameters():
                if param.grad == None:
                    tmp.append(zeros_like(param).view(-1).float())             
                    continue
                tmp.append(param.grad.view(-1).float())
                param.grad = None
        prev_grad = cat(tmp).to("cpu")
        
        all_reduce(prev_grad, op = ReduceOp.SUM, group=self.dp_group.group)
        
        tmp = split(prev_grad, self.len_sizes)
        with no_grad():
            for i, param in enumerate(self.net.parameters()):
                param.grad = tmp[i].view(self.sizes[i]).to(param.device).to(param.data.dtype) 
            clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()
        self.iteration += 1
        self.optimizer.step()
    
    def change_optimizer(self, net):
        self.net = net
        self.optimizer = Adam(net.parameters(),lr = lr)
        
        
        


