from simplellm.llama import LLamaFirstStage, LLamaStage, LLamaLastStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories, OpenWebText, RedPyjama # get our dataset
from simplellm.utils import State
from simplellm.losses import causalLLMLoss, perplexityLoss # our loss
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
import json
from time import time
from math import sqrt
rank = int(argv[3])
os.environ["MASTER_ADDR"] = "localhost"
world_size = int(argv[4])
os.environ["MASTER_PORT"] = "29500"
h_failure_probability = int(argv[5])/100
dist.init_process_group("gloo", rank=rank, world_size=world_size)
start_iter = int(argv[7]) if len(argv) > 7 else 0
with open(argv[6],"r") as fd:
    config = json.load(fd)

dmodel = config["dmodel"]
num_heads = config["num_heads"]
n_layers_per_stage = config["n_layers_per_stage"]
n_stages = config["n_stages"]
seq_l = config["n_stages"]
batch_size = config["batch_size"]
lr_scale = config["lr_scale"]
mb_count = config["mb_count"]
validation_amount = config["validation"]
checkpoint_mode = argv[1]
device = argv[2]
# make the tokenizer
tokenizer = SPTokenizer()
world_data_size = world_size
rank_data_size = rank
if config["architecture"] == "LLaMa":
    s0 = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=0, ctx_size=seq_l,padding_idx=tokenizer.pad_id,de_embed=True)
    stages = [s0]

    # Make the stages:
    for _ in range(n_stages):
        stages.append(LLamaStage(dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id))

means = [0 for _ in range(len(stages))]
stds = [1 for _ in range(len(stages))]
prev_gradient_norm = [1 for _ in range(len(stages))]

if config["dataset"] == "OpenWebText":
    ds = OpenWebText(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=start_iter*(world_size*mb_count) + validation_amount*2)
    validation_dataset = OpenWebText(tokenizer,batch_size=batch_size, seq_l=seq_l)
elif config["dataset"] == "RedPyjamas":
    splits = ["arxiv","c4","common_crawl","wikipedia"]
    s = splits[rank % 4]
    world_data_size = world_size // 4
    rank_data_size = rank // 4
    ds = RedPyjama(tokenizer,batch_size=batch_size, seq_l=seq_l,group=s,skip=start_iter*(world_size*mb_count) + validation_amount*2)
    validation_dataset = RedPyjama(tokenizer,batch_size=batch_size, seq_l=seq_l,group=s)


# we can iterate the dataset with:
iter_ds = iter(ds)

optimizers = []
optimizer_checkpoints = []
for i in range(len(stages)):
    optimizers.append(Adam(stages[i].parameters(),lr=3e-4))

checkpoints = []

# used for dp communication
vls = []
once = True
for s in stages:
    sizes = []
    len_sizes = []
    
    for param in s.parameters():
        sizes.append(param.shape)
        len_sizes.append(len(param.view(-1)))
    vls.append((sizes,len_sizes))
    if once:
        once = False
        print("Bytes in first stage",sum(len_sizes) * 8)
print("Bytes in other stages", sum(vls[-1][1]) * 8)


# Convert hourly probability to iter_probability
iter_vs = iter(validation_dataset)
t1 = time()
mb_size = batch_size * dmodel * seq_l * 8

for _ in range(mb_count): 
    with torch.no_grad():
        x = next(iter_vs)
        x = x.to(device)
        target = x.clone().detach()
        for i,s in enumerate(stages):
            if i == 0:
                x = s.embed(x)
            else:
                x = s(x)
                
                    
        x = stages[0].forward_end(x)
t1 = time() - t1
# 80 MB/s or 640Mb/s
t1 += len(stages)*mb_size / (0.08*1024**3)
t1 += len(stages)*0.3 # account for trailing and other delays
print("time for F to B",t1 * 2.5) # backwards is a bit slower
print("time for dp", (world_size - 1) * sum(vls[-1][1]) * 8 / (0.08*1024**3))
total_time = t1 * 2.5 + (world_size - 1) * sum(vls[-1][1]) * 8 / (0.08*1024**3)
print("total time per iteration ", total_time)
iter_success_probability = sqrt((100 - h_failure_probability)/100)
print("Iteration failure probability ", 1 - iter_success_probability)
for itr in range(50_000):
    for optim in optimizers:
        optim.zero_grad()
    
    # checkpoint:
    if checkpoint_mode in ["whole_model", "one"]:
        optimizer_checkpoints.clear()
        for optim in optimizers:
            optimizer_checkpoints.append(deepcopy(optim.state_dict()))
        checkpoints.clear()
        for s in stages:
            checkpoints.append(deepcopy(s.state_dict()))

    this_round_loss = 0
    failures = [-1 for _ in range(len(stages))]
    for s in range(len(stages)):
        if s == 0:
            continue
        can_fail = random.random() > iter_success_probability
        if can_fail:
            failures[s] = random.randint(0,mb_count-1)
    
    
    for mbid in range(mb_count): 
        x = None
        for idx in range(world_data_size):
            if idx == rank_data_size:
                x = next(iter_ds)
                x = x.to(device)
            else:
                next(iter_ds)
        target = x.clone().detach()
        can_fail = True
        for i,s in enumerate(stages):
            if  mbid == failures[i]:
                print("failure",itr,mbid,i)
                if checkpoint_mode == "ours-naive":
                    if i == 1:
                        selector = i + 1
                    else:
                        selector = i - 1
                    s.load_state_dict(deepcopy(stages[selector].state_dict()))
                    optimizers[i] = Adam(s.parameters(),lr = lr_scale*3e-4)
                        
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
        
        for p in s.parameters():
            if p.grad == None:
                tmp.append(zeros_like(p.data).view(-1))   
                continue
            tmp.append(p.grad.view(-1))
               
        tmp = cat(tmp)
        prev_gradient_norm[i] = torch.linalg.vector_norm(tmp).item() + 0.00001
        torch.nn.utils.clip_grad_norm_(s.parameters(),max_norm=1.0)
    
    for optim in optimizers:
        optim.step()   
    if itr % 100 == 0 and rank == 0:
        print("SAVING ITERATION",itr)
        for i,s in enumerate(stages):
            torch.save(s.state_dict(), f"mdl_{i}.pth") 
        print("SAVED")
    
    optim.zero_grad()
    if itr % 100 == 0:
        perplxities = []
        iter_vs = iter(validation_dataset)
        for _ in range(validation_amount): 
            with torch.no_grad():
                x = next(iter_vs)
                x = x.to(device)
                target = x.clone().detach()
                for i,s in enumerate(stages):
                    if i == 0:
                        x = s.embed(x)
                    else:
                        x = s(x)
                x = stages[0].forward_end(x)
                loss = perplexityLoss(x,target)
                perplxities.append(loss.item())
        print("VALIDATION LOSS",itr,sum(perplxities)/len(perplxities))
            
    dist.barrier()

    
    cuda.empty_cache()



