from torch.distributed import new_group
from torch.distributed import init_process_group
import os
from time import sleep
def initialise_communication(partitions, pid, addr, world_size, delay_map):
    
    return DP_Group(partitions,pid, delay_map)

class DP_Group(object):
    def __init__(self, partitions, pid):
        self.group = None
        self.pid = pid
        self.delay_map = delay_map
        self.partition = 0
        self.g_size = 0
        self.in_group = 0
        self.total_partitions = len(partitions)
        
        
        for idx,p in enumerate(partitions):
            if pid in p:

                
                for idx2,pid2 in enumerate(p):
                    if pid2 == pid:
                        self.in_group = idx2
                        continue
                self.g_size = len(p)
                self.partition  = idx
            
    
   

    