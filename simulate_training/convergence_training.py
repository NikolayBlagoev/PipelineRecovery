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
from torch.optim import AdamW, Adam
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

h_failure_probability = int(argv[2])

start_iter = int(argv[4]) if len(argv) > 4 else 0
with open(argv[3],"r") as fd:
    config = json.load(fd)
checkpoint_mode = argv[1]
gamma = 1e-3 if "regularize" in checkpoint_mode else 0
init_lr = config["lr"]
if "regularize" in checkpoint_mode:
    checkpoint_mode = checkpoint_mode[:-len("-regularize")]
    print("NEW CHECKPOINT MODE", checkpoint_mode, checkpoint_mode == "ours-grad-avg")
def custom_loss(net1,net2,net3,itr):
    return
    l = 0
    count = 0
    if net3 == None:
        for p1,p2 in zip(net1.parameters(), net2.parameters()):
            
            
            p1.grad -=  max(0.25,math.exp(-itr/10_000)) * 0.5 * gamma * (p1 - p2)
    else:

        for p1,p2,p3 in zip(net1.parameters(), net2.parameters(), net3.parameters()):
            
            p1.grad -=  max(0.25,math.exp(-itr/10_000))*gamma * (p1 - (0.5*p2 + 0.5*p3))
            # l += 1-F.cosine_similarity(p1.view(-1), (0.5*p2 + 0.5*p3).view(-1),dim=0)
    


            
dmodel = config["dmodel"]
num_heads = config["num_heads"]
n_layers_per_stage = config["n_layers_per_stage"]
n_stages = config["n_stages"]
seq_l = config["seq_l"]
batch_size = config["batch_size"]
lr_scale = config["lr_scale"]
mb_count = config["mb_count"]
validation_amount = config["validation"]
max_iterations = config["max_iterations"] 

num_warmup_steps = 500
num_training_steps = max_iterations
max_iterations = max_iterations - start_iter
num_cycles = 0.5
def lr_lambda(current_step: int) -> float:
    current_step += start_iter
    
    if current_step < num_warmup_steps:
        return current_step / max(1, num_warmup_steps)
    if current_step > 10000:
        return 1/10
    progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
    )

    cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
    )
    return max(1/10, 1/10 + cosine_lr_multiple)

    
# make the tokenizer
def make_optim(params,lr,itr = 0):
    return LambdaLR(AdamW(params, lr, betas=(0.9, 0.9), weight_decay=0.01),lr_lambda)


if config["architecture"] == "LLaMa":
    tokenizer = SPTokenizer()
    torch.manual_seed(34107)
    s0 = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                    device="cuda:0", n_layers=0, ctx_size=seq_l,padding_idx=tokenizer.pad_id,de_embed=True)
    
    stages = [s0]

    # Make the stages:
    
    for k in range(n_stages):
        # torch.manual_seed(34107)
        stages.append(LLamaStage(dmodel=dmodel,num_heads=num_heads,
                    device=f"cuda:{(1 + k)//2}", n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id))
elif config["architecture"] == "GPT":
    tokenizer = GPTTokenizer()
    torch.manual_seed(34107)
    s0 = GPTFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads, device="cuda:0",
                            n_layers=0, ctx_size=seq_l, padding_idx=tokenizer.pad_id, de_embed=True,dropout_prob=0)
    stages = [s0]

    # Make the stages:
    for k in range(n_stages):
        stages.append(GPTStage(dmodel=dmodel,num_heads=num_heads,
                    device=f"cuda:{(1 + k)//2}", n_layers=n_layers_per_stage, ctx_size=seq_l,dropout_prob=0))

# print(len(stages))
if start_iter > 0:
    for i,s in enumerate(stages):
        s.load_state_dict(torch.load(f"mdl_{i}.pth",map_location=f"cuda:{i//2}"))

optimizers = []
optimizer_checkpoints = []
for i in range(len(stages)):
    optimizers.append(make_optim(stages[i].parameters(),lr=init_lr))

if start_iter > 0:
    for i in range(len(stages)):
        optimizers[i].load_state_dict(torch.load(f"optim_{i}.pth",map_location=f"cuda:{i//2}"))
means = [0 for _ in range(len(stages))]
stds = [1 for _ in range(len(stages))]
prev_gradient_norm = [1 for _ in range(len(stages))]
prev_gradient_norm_inf = [1 for _ in range(len(stages))]
if config["dataset"] == "OpenWebText":
    ds = OpenWebText(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=start_iter*(1*mb_count) + validation_amount*2)
    validation_dataset = OpenWebText(tokenizer,batch_size=16, seq_l=seq_l)
elif config["dataset"] == "RedPyjamas":
    ds = RedPyjamav2(tokenizer,batch_size=batch_size, seq_l=seq_l,name="default",skip=start_iter*(1*mb_count) + validation_amount*2)
    validation_dataset = RedPyjamav2(tokenizer,batch_size=1, seq_l=seq_l,name="default")
elif config["dataset"] == "TinyStories":
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=start_iter*(1*mb_count))
    validation_dataset = TinyStories(tokenizer,batch_size=16, seq_l=seq_l, split="validation")



# we can iterate the dataset with:
iter_ds = iter(ds)



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
        x = x.to("cuda:0")
        target = x.clone().detach()
        for i,s in enumerate(stages):
            if i == 0:
                x = s.embed(x)
            else:
                # print(i,i//2)
                x = x.to(f"cuda:{i//2}")
                x = s(x)
                
        x = x.to("cuda:0")   
        x = stages[0].forward_end(x)
t1 = (time() - t1)
# 100 MB/s or ~800Mb/s
t1 += len(stages)*mb_size / (0.1*1024**3) # in reality you will do multiple waves
t1 = t1 * 2 # in reality you will do multiple waves
print("time for F to B",t1 * 2.5) # backwards is a bit slower
print("time for dp", (2*4 - 1) * sum(vls[-1][1]) * 8 / (0.2*1024**3))
total_time = t1 * 2.5 + (2*4 - 1) * sum(vls[-1][1]) * 8 / (0.2*1024**3) # on same cluster large devices
total_time *=  1 # synchronisation
print("total time per iteration ", total_time)
iterations_per_h = 60*60 / total_time 
print(60*60 / total_time, 100 - h_failure_probability)
error_term = [None for _ in range(n_stages)]
# iter_success_probability = ((100 - h_failure_probability)/100)**(total_time / 3600)
iter_success_probability = 1.0 - config[str(h_failure_probability)]
print("Iteration failure probability ", 1 - iter_success_probability)
for itr in range(max_iterations):
    try:
        for optim in optimizers:
            optim.optimizer.zero_grad()
        t1 = time()
        # checkpoint:
        if checkpoint_mode in ["whole_model", "one"]:
            optimizer_checkpoints.clear()
            for optim in optimizers:
                optimizer_checkpoints.append(deepcopy(optim.optimizer.state_dict()))
            checkpoints.clear()
            for s in stages:
                checkpoints.append(deepcopy(s.state_dict()))

        this_round_loss = 0
        failures = [-1 for _ in range(len(stages))]
        for s in range(len(stages)):
            stages[s].train()
            if s == 0:
                # holds embedding and dembedding
                continue
            can_fail = random.random() > iter_success_probability
            if can_fail:
                failures[s] = random.randint(0,mb_count-1)
                failures[s] = 0
        
        
        for mbid in range(mb_count): 
            x = None
            flg = True
            idx = 0
            while flg:
                flg = False
                try:
                    x = next(iter_ds)
                    x = x.to("cuda:0")
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
                        optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr,itr=itr)
                    elif checkpoint_mode == "ours-zero":
                        if i == 1:
                            selector = i + 1
                        else:
                            selector = i - 1
                        m1 = deepcopy(stages[selector].state_dict())
                        
                        s.load_state_dict(m1)
                        for ln in range(len(s.layers)):
                            s.layers[ln].mlp.down_proj.weight = torch.nn.parameter.Parameter(torch.zeros_like(s.layers[ln].mlp.down_proj.weight))
                    
                        optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr,itr=itr)
                        del m1
                    elif checkpoint_mode == "ours-random":
                        
                        stages[i] = LLamaStage(dmodel=dmodel,num_heads=num_heads,
                                device=device, n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id)
                        s = stages[i]
                        optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr,itr=itr)
                            
                    elif checkpoint_mode == "ours-grad-avg":
                        if i == len(stages)-1:
                            s.load_state_dict(deepcopy(stages[i-1].state_dict()))
                            optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr,itr=itr)
                            for optim in optimizers:
                                optim.optimizer.zero_grad()
                            
                            
                        elif i == 1: 
                            s.load_state_dict(deepcopy(stages[i+1].state_dict()))
                            optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr,itr=itr)
                            for optim in optimizers:
                                optim.optimizer.zero_grad()
                            
                            
                        else:
                            m1 = deepcopy(stages[i+1].state_dict())
                            m2 = deepcopy(stages[i-1].state_dict())
                            alpha = abs(prev_gradient_norm[i+1]) + 0.0001
                            beta = abs(prev_gradient_norm[i-1]) + 0.0001
                            if config["architecture"] == "LLaMa":
                                stages[i] = LLamaStage(dmodel=dmodel,num_heads=num_heads,
                                    device=f"cuda:{i//2}", n_layers=n_layers_per_stage, ctx_size=seq_l,padding_idx=tokenizer.pad_id)
                            else:
                                stages[i] = GPTStage(dmodel=dmodel,num_heads=num_heads,
                                    device=f"cuda:{i//2}", n_layers=n_layers_per_stage, ctx_size=seq_l,dropout_prob=0)
                            m3 = stages[i].state_dict()
                            for key in m1:
                                m3[key] = (alpha*m1[key] + beta*m2[key]) / (alpha + beta)
                            stages[i].load_state_dict(m3)
                            s = stages[i]
                            
                            optimizers[i] = make_optim(s.parameters(),lr = lr_scale*init_lr,itr=itr)
                            for optim in optimizers:
                                optim.optimizer.zero_grad()
                            
                            del m3
                            del m2
                            del m1
                    
                    elif checkpoint_mode == "one":
                        s.load_state_dict(deepcopy(checkpoints[i]))
                        optimizers[i] = make_optim(s.parameters(),lr = init_lr,itr=itr)
                        optimizers[i].load_state_dict(deepcopy(optimizer_checkpoints[i]))
                    elif checkpoint_mode == "whole_model":
                        for idx,s2 in enumerate(stages):
                            stages[idx].load_state_dict(deepcopy(checkpoints[idx]))
                            optimizers[idx] = make_optim(stages[idx].parameters(),lr = init_lr,itr=itr)
                            optimizers[idx].load_state_dict(deepcopy(optimizer_checkpoints[idx]))
                    elif checkpoint_mode == "no_failure":
                        can_fail = False
                    

                if i == 0:
                    x = s.embed(x)
                else:
                    x = x.to(f"cuda:{i//2}")
                    x = s(x)
            x = x.to("cuda:0")
            x = stages[0].forward_end(x)
            
            loss = causalLLMLoss(x,target,tokenizer.vocab_size)
            
            loss = loss / mb_count
            
            this_round_loss += loss.item()
            
            loss.backward()
        print(itr,this_round_loss)
        
            
       

        # Sync weights
        # for idx,s in enumerate(stages):
        #     tmp = []
        #     for param in s.parameters():
        #         if param.grad == None:
        #             tmp.append(torch.zeros_like(param,device="cpu").view(-1))                      
        #             continue
        #         tmp.append(param.grad.view(-1))
        #         param.grad = None
        #     prev_grad = torch.cat(tmp).to("cpu")
        #     dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
        #     tmp = torch.split(prev_grad, vls[idx][1])
        #     for i, param in enumerate(s.parameters()):
        #         param.grad = tmp[i].view(vls[idx][0][i]).to(device)/world_size # average
        stages_tmps = []
        for i,s in enumerate(stages):
            tmp = []
            tmp2 = []
            torch.nn.utils.clip_grad_norm_(s.parameters(),max_norm=1.0)
            for p in s.parameters():
                if p.grad == None:
                    tmp.append(zeros_like(p.data).view(-1))   
                    tmp2.append(zeros_like(p.data).view(-1))
                    continue
                tmp.append(p.grad.view(-1))
                tmp2.append(p.data.view(-1))
            
            tmp = cat(tmp)
            stages_tmps.append(cat(tmp2).to("cpu"))
            prev_gradient_norm[i] = prev_gradient_norm[i]*0 + 1.0*abs(torch.dot(tmp,tmp).item())
            prev_gradient_norm_inf[i] = prev_gradient_norm[i]*0 + 1.0*abs(torch.linalg.vector_norm(tmp,ord=float("inf")).item())
         
        for i,s in enumerate(stages):
            if i < 2 or i == len(stages) - 1:
                continue
            err = (prev_gradient_norm[i-1]*stages_tmps[i-1] + prev_gradient_norm[i+1]*stages_tmps[i+1])/(prev_gradient_norm[i-1] + prev_gradient_norm[i+1]) - stages_tmps[i]
            
            # if itr % 20 == 0:
            #     error_term[i] = None
            # if error_term[i] != None and itr % 20 != 0:
            #     err += error_term[i]
            err = torch.abs(err)
            
            print(itr,i, "MAX l2", torch.max(err).item())
            print(itr,i, "MEAN l2", torch.mean(err).item())
            err = (prev_gradient_norm_inf[i-1]*stages_tmps[i-1] + prev_gradient_norm_inf[i+1]*stages_tmps[i+1])/(prev_gradient_norm_inf[i-1] + prev_gradient_norm_inf[i+1]) - stages_tmps[i]
            err = torch.abs(err)
            print(itr,i, "MAX inf", torch.max(err).item())
            print(itr,i, "MEAN inf", torch.mean(err).item())
            tmp1 = stages_tmps[i+1] - stages_tmps[i-1]
            tmp2 = stages_tmps[i] - stages_tmps[i-1]
            coeff = torch.dot(tmp1,tmp2)/torch.dot(tmp2,tmp2)
            err = coeff * tmp1 - stages_tmps[i]
            err = torch.abs(err)
            print(itr,i, "MAX dot", torch.max(err).item())
            print(itr,i, "MEAN dot", torch.mean(err).item())
        
        for optim in optimizers:
            optim.optimizer.step()
            optim.step(itr) 
            
        if itr % 100 == 0 :
            print("SAVING ITERATION",itr)
            for i,s in enumerate(stages):
                torch.save(s.state_dict(), f"mdl_{i}.pth") 
                torch.save(optimizers[i].state_dict(), f"optim_{i}.pth")
            print("SAVED")
        
        
        if itr % 500 == 0:
            perplxities = []
            normal_loss = []
            iter_vs = iter(validation_dataset)
            for _ in range(validation_amount): 
                with torch.no_grad():
                    x = next(iter_vs)
                    x = x.to("cuda:0")
                    target = x.clone().detach()
                    for i,s in enumerate(stages):
                        s.eval()
                        if i == 0:
                            x = s.embed(x)
                        else:
                            x = x.to(f"cuda:{i//2}")
                            x = s(x)
                    x = x.to("cuda:0")
                    x = stages[0].forward_end(x)
                    loss = perplexityLoss(x,target)
                    perplxities.append(loss.item())
                    loss = causalLLMLoss(x,target,tokenizer.vocab_size)
                    normal_loss.append(loss.item())
            print("VALIDATION LOSS",itr,sum(perplxities)/len(perplxities))
            print("NORMAL LOSS",itr,sum(normal_loss)/len(normal_loss))
                
        # dist.barrier()
        print("time:",time()-t1)
        
        cuda.empty_cache()
    except StopIteration:
        iter_ds = iter(ds)
    except Exception:
        print(traceback.format_exc())




