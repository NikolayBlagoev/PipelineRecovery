from simplellm.llama import LLamaFirstStage, LLamaStage, LLamaLastStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import OpenWebText, Wikipedia_Dataset # get our dataset
from simplellm.utils import State
from simplellm.losses import causalLLMLoss # our loss
from copy import deepcopy
from sys import argv
import random
random.seed(42)
State.set_seed(42)
from torch.optim import Adam
import torch.nn.functional as F
from torch import save, cuda, zeros_like, cat, mean, std
import torch
import torch.distributed as dist
import os

rank = int(argv[3])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 8
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)


dmodel = 1024
num_heads = 16
n_layers_per_stage = 4
n_stages = 6
seq_l = 1024
batch_size = 16
lr_scale = 1
mb_count = 4

checkpoint_mode = argv[1]
device = argv[2]
# make the tokenizer
tokenizer = SPTokenizer()
# make the model

s0 = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=0, ctx_size=seq_l,padding_idx=tokenizer.pad_id,de_embed=True)
stages = [s0]
for _ in range(n_stages):
    stages.append(LLamaStage(dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id))

means = [0 for _ in range(len(stages))]
stds = [1 for _ in range(len(stages))]
prev_gradient_norm = [1 for _ in range(len(stages))]
ds = OpenWebText(tokenizer,batch_size=batch_size, seq_l=seq_l)

# we can iterate the dataset with:
iter_ds = iter(ds)
optimizers = []
optimizer_checkpoints = []

for i in range(len(stages)):
    optimizers.append(Adam(stages[i].parameters(),lr=3e-4))

checkpoints = []
learning_rates = []


vls = []
for s in stages:
    sizes = []
    len_sizes = []
    for param in s.parameters():
        sizes.append(param.shape)
        len_sizes.append(len(param.view(-1)))
    vls.append((sizes,len_sizes))
for itr in range(10_000):
    for optim in optimizers:
        optim.zero_grad()
    if checkpoint_mode in ["whole_model", "one"]:
        optimizer_checkpoints.clear()
        for optim in optimizers:
            optimizer_checkpoints.append(deepcopy(optim.state_dict()))
        checkpoints.clear()
        for s in stages:
            checkpoints.append(deepcopy(s.state_dict()))

    this_round_loss = 0
    can_fail = random.random() < 1/50
    failing_node = random.randint(1,len(stages)-1)
    
    for _ in range(mb_count): 
        x = None
        for idx in range(world_size):
            if idx == rank:
                x = next(iter_ds)
                x = x.to(device)
            else:
                next(iter_ds)
        target = x.clone().detach()
        for i,s in enumerate(stages):
            if  (i == failing_node or checkpoint_mode == "whole_model") and can_fail:
                print("failure",failing_node)
                can_fail = False
                if checkpoint_mode == "ours-naive":
                    if i == 1:
                        selector = i + 1
                    else:
                        selector = i - 1
                    s.load_state_dict(deepcopy(stages[selector].state_dict()))
                    optimizers[i] = Adam(s.parameters(),lr = 3e-4)
                        

                elif checkpoint_mode == "ours-grad-avg":
                    if i == len(stages)-1:
                        
                        s.load_state_dict(deepcopy(stages[i-1].state_dict()))
                        optimizers[i] = Adam(s.parameters(),lr = lr_scale*3e-4)
                    elif i == 1: 
                        s.load_state_dict(deepcopy(stages[i+1].state_dict()))
                        optimizers[i] = Adam(s.parameters(),lr = lr_scale*3e-4)
                    else:
                        m1 = deepcopy(stages[i+1].state_dict())
                        m2 = deepcopy(stages[i-1].state_dict())
                        alpha = abs(prev_gradient_norm[i+1]) + 0.0001
                        beta = abs(prev_gradient_norm[i-1]) + 0.0001
                        m3 = s.state_dict()
                        for key in m1:
                            m3[key] = (alpha*m1[key] + beta*m2[key]) / (alpha + beta)
                        s.load_state_dict(m3)
                        
                        optimizers[i] = Adam(s.parameters(),lr = lr_scale*3e-4)
                        del m3
                        del m2
                        del m1
                
                elif checkpoint_mode == "one":
                    s.load_state_dict(deepcopy(checkpoints[i]))
                    optimizers[i] = Adam(s.parameters(),lr = 3e-4)
                    optimizers[i].load_state_dict(deepcopy(optimizer_checkpoints[i]))
                elif checkpoint_mode == "whole_model":
                    for idx,s2 in enumerate(stages):
                        stages[idx].load_state_dict(deepcopy(checkpoints[idx]))
                        optimizers[idx] = Adam(stages[idx].parameters(),lr = 3e-4)
                        optimizers[idx].load_state_dict(deepcopy(optimizer_checkpoints[idx]))
                elif checkpoint_mode == "no_failure":
                    can_fail = False
                

            if i == 0:
                x = s.embed(x)
            else:
                x = s(x)
        x = stages[0].forward_end(x)
        
        loss = causalLLMLoss(x,target,tokenizer.vocab_size)
        loss = loss / mb_count
        can_fail = False
        this_round_loss += loss.item()
        
        loss.backward()
    print(itr,this_round_loss)
    dist.barrier() # wait for everyone
    for idx,s in enumerate(stages):
        tmp = []
        for param in s.parameters():
            if param.grad == None:
                tmp.append(torch.zeros_like(param,device="cpu").view(-1))                      
                continue
            tmp.append(param.grad.view(-1))
            param.grad = None
        prev_grad = torch.cat(tmp).to("cpu")
        dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
        tmp = torch.split(prev_grad, vls[idx][1])
        for i, param in enumerate(s.parameters()):
            param.grad = tmp[i].view(vls[idx][0][i]).to(device)/world_size # average
    for i,s in enumerate(stages):
            tmp = []
            w_tmp = []
            for p in s.parameters():
                if p.grad == None:
                    tmp.append(zeros_like(p.data).view(-1))
                    
                    continue
                tmp.append(p.grad.view(-1))
               
            tmp = cat(tmp)
            prev_gradient_norm[i] = torch.linalg.vector_norm(tmp).item() + 0.00001
    
    for optim in optimizers:
        optim.step()   

    
    cuda.empty_cache()



