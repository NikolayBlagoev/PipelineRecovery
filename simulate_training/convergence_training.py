from simplellm.llama import LLamaFirstStage, LLamaStage, LLamaLastStage # get our models
from simplellm.gpt import GPTFirstStage, GPTStage
from simplellm.tokenizers import SPTokenizer, GPTTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories, OpenWebText, RedPyjamav2 # get our dataset
from simplellm.utils import State
from simplellm.losses import causalLLMLoss, perplexityLoss # our loss
from copy import deepcopy
from sys import argv
import random
random.seed(42)
State.set_seed(42)
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch import save, cuda, zeros_like, cat, mean, std
import torch
import torch.distributed as dist
import traceback
import os
import json
from time import time
from math import sqrt
import math
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
max_iterations = config["max_iterations"] 
init_lr = config["lr"]
checkpoint_mode = argv[1]
device = argv[2]
num_warmup_steps = 2000
num_training_steps = max_iterations
max_iterations = max_iterations - start_iter
num_cycles = 0.5
def lr_lambda(current_step: int) -> float:
    # linear warmup phase
    current_step += start_iter
    if current_step < num_warmup_steps:
        return current_step / max(1, num_warmup_steps)

    # cosine
    progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
    )

    cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
    )
    return max(0.0, cosine_lr_multiple)

    
# make the tokenizer
def make_optim(params,lr):
    return Adam(params, lr)

world_data_size = world_size
rank_data_size = rank
if config["architecture"] == "LLaMa":
    tokenizer = SPTokenizer()
    s0 = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=0, ctx_size=seq_l,padding_idx=tokenizer.pad_id,de_embed=True)
    
    stages = [s0]

    # Make the stages:
    for _ in range(n_stages):
        stages.append(LLamaStage(dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id))
elif config["architecture"] == "GPT":
    tokenizer = GPTTokenizer()
    s0 = GPTFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads, device=device,
                            n_layers=0, ctx_size=seq_l, padding_idx=tokenizer.pad_id, de_embed=True)
    stages = [s0]

    # Make the stages:
    for _ in range(n_stages):
        stages.append(GPTStage(dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=n_layers_per_stage, ctx_size=seq_l))
if start_iter > 0:
    for i,s in enumerate(stages):
        s.load_state_dict(torch.load(f"mdl_{i}.pth",map_location=device))

means = [0 for _ in range(len(stages))]
stds = [1 for _ in range(len(stages))]
prev_gradient_norm = [1 for _ in range(len(stages))]

if config["dataset"] == "OpenWebText":
    ds = OpenWebText(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=start_iter*(world_size*mb_count) + validation_amount*2)
    validation_dataset = OpenWebText(tokenizer,batch_size=16, seq_l=seq_l)
elif config["dataset"] == "RedPyjamas":
    ds = RedPyjamav2(tokenizer,batch_size=batch_size, seq_l=seq_l,name="default",skip=start_iter*(world_data_size*mb_count) + validation_amount*2)
    validation_dataset = RedPyjamav2(tokenizer,batch_size=16, seq_l=seq_l,name=s)
elif config["dataset"] == "TinyStories":
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=start_iter*(world_size*mb_count))
    validation_dataset = TinyStories(tokenizer,batch_size=16, seq_l=seq_l, split="validation")



# we can iterate the dataset with:
iter_ds = iter(ds)

optimizers = []
optimizer_checkpoints = []
for i in range(len(stages)):
    optimizers.append(make_optim(stages[i].parameters(),lr=init_lr))

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
print("time for dp", (2*world_size - 1) * sum(vls[-1][1]) * 8 / (0.08*1024**3))
total_time = t1 * 2.5 + (2*world_size - 1) * sum(vls[-1][1]) * 8 / (0.08*1024**3)
total_time *= 1.1 # synchronisation
print("total time per iteration ", total_time)
iterations_per_h = 60*60 / total_time 
iter_success_probability = ((100 - h_failure_probability)/100)**(1/iterations_per_h)
print("Iteration failure probability ", 1 - iter_success_probability)
for itr in range(max_iterations):
    try:
        for optim in optimizers:
            optim.zero_grad()
        t1 = time()
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
            stages[s].train()
            if s == 0:
                continue
            can_fail = random.random() > iter_success_probability
            if can_fail:
                failures[s] = random.randint(0,mb_count-1)
        
        
        for mbid in range(mb_count): 
            x = None
            flg = True
            idx = 0
            while flg:
                flg = False
                try:
                    while idx < rank_data_size:
                        if idx == rank_data_size:
                            x = next(iter_ds)
                            x = x.to(device)
                        else:
                            next(iter_ds)
                        idx+=1
                except StopIteration:
                    flg = True
                    iter_ds = iter(ds)
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
                        optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr)
                        
                            
                    elif checkpoint_mode == "ours-grad-avg":
                        if i == len(stages)-1:
                            
                            s.load_state_dict(deepcopy(stages[i-1].state_dict()))
                            optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr)
                        elif i == 1: 
                            s.load_state_dict(deepcopy(stages[i+1].state_dict()))
                            optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr)
                        else:
                            m1 = deepcopy(stages[i+1].state_dict())
                            m2 = deepcopy(stages[i-1].state_dict())
                            alpha = abs(prev_gradient_norm[i+1]) + 0.0001
                            beta = abs(prev_gradient_norm[i-1]) + 0.0001
                            m3 = s.state_dict()
                            for key in m1:
                                m3[key] = (alpha*m1[key] + beta*m2[key]) / (alpha + beta)
                            s.load_state_dict(m3)
                            
                            optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr)
                            del m3
                            del m2
                            del m1
                    
                    elif checkpoint_mode == "one":
                        s.load_state_dict(deepcopy(checkpoints[i]))
                        optimizers[i] = make_optim(s.parameters(),lr = init_lr)
                        optimizers[i].load_state_dict(deepcopy(optimizer_checkpoints[i]))
                    elif checkpoint_mode == "whole_model":
                        for idx,s2 in enumerate(stages):
                            stages[idx].load_state_dict(deepcopy(checkpoints[idx]))
                            optimizers[idx] = make_optim(stages[idx].parameters(),lr = init_lr)
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

        # Sync weights
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
            optim.step()   
            
        if itr % 100 == 0 and rank == 0:
            print("SAVING ITERATION",itr)
            for i,s in enumerate(stages):
                torch.save(s.state_dict(), f"mdl_{i}.pth") 
                torch.save(optimizers[i].state_dict(), f"optim_{i}.pth")
            print("SAVED")
        
        
        if itr % 100 == 0:
            perplxities = []
            normal_loss = []
            iter_vs = iter(validation_dataset)
            for _ in range(validation_amount): 
                with torch.no_grad():
                    x = next(iter_vs)
                    x = x.to(device)
                    target = x.clone().detach()
                    for i,s in enumerate(stages):
                        s.eval()
                        if i == 0:
                            x = s.embed(x)
                        else:
                            x = s(x)
                    x = stages[0].forward_end(x)
                    loss = perplexityLoss(x,target)
                    perplxities.append(loss.item())
                    loss = causalLLMLoss(x,target,tokenizer.vocab_size)
                    normal_loss.append(loss.item())
            print("VALIDATION LOSS",itr,sum(perplxities)/len(perplxities))
            print("NORMAL LOSS",itr,sum(normal_loss)/len(normal_loss))
                
        dist.barrier()
        print("time:",time()-t1)
        
        cuda.empty_cache()
    except StopIteration:
        iter_ds = iter(ds)
    except Exception:
        print(traceback.format_exc())




