from dataclasses import dataclass
import os
import random
from typing import Callable, Dict
from deccom.peers.peer import Peer
from deccom.protocols.abstractprotocol import AbstractProtocol
from deccom.protocols.wrappers import *
from datetime import datetime
import asyncio
from traceback import print_exception, format_exc
from .llm_subp import *
from time import sleep, time
from deccom.cryptofuncs.hash import SHA256

@dataclass
class LocalPeer:
    peer: Peer
    stage: int
    meshid: int
    has_weights: bool
'''
Responsible for communication
'''
class PPProtocl(AbstractProtocol):
    required_lower = AbstractProtocol.required_lower + \
        ["find_peer", "get_peer", "get_peers", "connected_callback","disconnected_callback"]
    FORWARD_FLAG = int.from_bytes(b'\x01', byteorder="big")
    BACK_FLAG = int.from_bytes(b'\x02', byteorder="big")
    AGGREGATE_FLAG = int.from_bytes(b'\x03', byteorder="big")
    MODEL_REQUEST_FLAG = int.from_bytes(b'\x04', byteorder="big")
    MODEL_RESPONSE_FLAG = int.from_bytes(b'\x05', byteorder="big")
    GRADIENTS_FLAG = int.from_bytes(b'\x06', byteorder="big")
    INTRODUCTION = int.from_bytes(b'\x07', byteorder="big")
    FORGET_ME = int.from_bytes(b'\x08', byteorder="big")

    
    def __init__(self, stage, meshid, MAX_STAGE, fail_at, crash_callback, queue_in: Queue, queue_out: Queue, subprocess:Process, submodule=None, callback: Callable[[tuple[str, int], bytes], None] = lambda : ..., 
                    MAX_SEND = 6, stage_size = 0, has_weights = True):
        
        super().__init__(submodule, callback)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.subprocess = subprocess
        self.disconnected_callback = lambda *args : ...
        self.connected_callback = lambda *args : ...
        self.crash_callback = crash_callback

        
        self.stage = stage
        self.MAX_STAGE = MAX_STAGE
        self.MAX_SEND = MAX_SEND
        self.mb_send = 0
        self.iteration = 0
        self.fail_at = fail_at
        self.meshid = meshid
        self.peers_without_weights = 0
        self.has_weights = has_weights
        self.stage_size = stage_size
        self.peers: Dict[int,LocalPeer] = {}
        self.deferred_tasks = []
        self.requested = False
        self.store_other = None
        self.same_stage_without_weights = 0
        self.request_lr = [False,False]
        
        self.send_receives = dict()
        
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None

    @bindto("remove_peer")
    def _lower_remove_peer(self, addr: tuple[str, int], node_id: bytes)->None:
        return None
    
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None

    @bindto("ping")
    async def send_ping(self, addr, success, error, dt = 10):
        return None

    async def start_iteration(self):
        await asyncio.sleep(10)
        
        for b in range(3):
            tag = b
            self.mb_send += 1
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"QUEUEIN MB {tag}\n")
            nxt = None if self.stage == self.MAX_STAGE-1 else int(self.peers[self.meshid + self.stage_size].peer.pub_key)
            
            self.queue_out.put(Start(tag,nxt,self.peer.pub_key), True)

    

    async def start(self, p: Peer):
        await super().start(p)
        
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            log.write(f"===={self.peer.pub_key} {self.stage} STARTING===\n")
        
        loop = asyncio.get_event_loop() 
        self.queue_reader = loop.create_task(self.read_from_queue())
        if self.stage == 0:
            await asyncio.sleep(2)
            loop.create_task(self.start_iteration())
        

    async def announce_end(self):
        self.mb_send = 0
        self.send_receives.clear()
        self.received_aggregates += 1
        self.queue_out.put(Aggregate(0), True)
        msg = bytearray()
        msg += PPProtocl.AGGREGATE_FLAG.to_bytes(1,byteorder="big")

        
        for p in self.peers.values():
            pub_key = str(p.peer.pub_key)
            if pub_key == self.peer.pub_key:
                continue
            p = await self._lower_find_peer(SHA256(pub_key))
                                
            await self.send_datagram(msg, p.addr)
            

    async def read_from_queue(self):
        while self.started:
            while self.queue_in.empty() and self.started:
                await asyncio.sleep(0.1)
            if not self.started:
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"====CLOSING???===\n")
                return
            task = self.queue_in.get(True)
            try:
                if isinstance(task, Forward):
                    
                    
                    msg = bytearray()
                    msg += PPProtocl.FORWARD_FLAG.to_bytes(1,byteorder="big")
                    msg += task.tag.to_bytes(4,byteorder="big")
                    msg += int(self.peer.pub_key).to_bytes(4,byteorder="big")
                    msg += task.originator.to_bytes(4,byteorder="big")
                    msg += task.data
                    sndto = str(task.to)
                   
                    p = await self._lower_find_peer(SHA256(sndto))
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.send_stream(p.id_node,msg))
                    with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                            log.write(f"Sending FORWARD {sndto} {task.tag} {time()}\n")
                    # await 
                    continue
                elif isinstance(task, Loss):
                    
                    msg = bytearray()
                    msg += PPProtocl.BACK_FLAG.to_bytes(1,byteorder="big")
                    msg += task.tag.to_bytes(4,byteorder="big")
                    msg += int(self.peer.pub_key).to_bytes(4,byteorder="big")
                    msg += task.originator.to_bytes(4,byteorder="big")
                    msg += task.data
                    sndto = str(task.to)
                    p = await self._lower_find_peer(SHA256(sndto))
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.send_stream(p.id_node,msg))
                    
                elif isinstance(task, Backward):
                    if self.stage != 0:
                        msg = bytearray()
                        msg += PPProtocl.BACK_FLAG.to_bytes(1,byteorder="big")
                        msg += task.tag.to_bytes(4,byteorder="big")
                        msg += int(self.peer.pub_key).to_bytes(4,byteorder="big")
                        msg += task.originator.to_bytes(4,byteorder="big")

                        msg += task.data
                        sndto = str(task.to)
                        p = await self._lower_find_peer(SHA256(sndto))
                        loop = asyncio.get_event_loop()
                        loop.create_task(self.send_stream(p.id_node,msg))
                        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                            log.write(f"Sending BACKWARD {sndto} {task.tag} {time()} {len(msg)}\n")
                        
                    else:
                        
                        if self.mb_send < self.MB_SEND_COUNT:
                            
                            
                            tag = task.tag
                            self.mb_send += 1
                            nxt = None if self.stage == self.MAX_STAGE-1 else int(self.peers[self.meshid + self.stage_size].peer.pub_key)
                            self.queue_out.put(Start(tag,nxt,int(self.peer.pub_key)), True)
                        elif self.mb_send == self.MB_SEND_COUNT and self.memory == self.MAX_MEM:
                            
                            await self.announce_end()
                            continue
                        elif self.mb_send > self.MB_SEND_COUNT:
                            raise Exception(f"Too many microbatches have been sent? {self.memory} {self.MAX_MEM} {self.mb_send} {self.MB_SEND_COUNT}")
                        
                        continue

                    continue
                
                elif isinstance(task, Gradients):
                    msg = bytearray()
                    msg += PPProtocl.GRADIENTS_FLAG.to_bytes(1,byteorder="big")
                    msg += task.data
                    for p in self.peers.values():
                        if p.stage == self.stage:
                            loop = asyncio.get_event_loop()
                            loop.create_task(self.send_stream(p.peer.id_node,msg))
                elif isinstance(task, SendWeights):
                    msg = bytearray()
                    msg += PPProtocl.MODEL_RESPONSE_FLAG.to_bytes(1,byteorder="big")
                    msg += task.data
                    sndto = str(task.frm)
                    p = await(self._lower_find_peer(SHA256(sndto)))
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.send_stream(p.id_node,msg))




                elif isinstance(task, Aggregate):
                    
                    self.iteration += 1
                    if self.iteration == self.fail_at:
                        msg = bytearray()
                        msg += PPProtocl.FORGET_ME.to_bytes(1,byteorder="big")
                        msg += self.peer.id_node
                        loop = asyncio.get_event_loop()
                        for p in self.peers.values():
                            
                            loop.create_task(self.send_datagram(msg, p.peer.addr))
                        loop.create_task(self.stop())
                        return
                    if self.stage != 0:
                        continue
                    
                    self.mb_send = 0
                    loop = asyncio.get_event_loop() 
                    loop.create_task(self.start_iteration())
                    
            except Exception as e:
                with open(f'log{self.peer.pub_key}.txt', 'a') as f:
                    f.write(str(e))
                    f.write("!!!!!!!!!!!!!!!\n")
                    f.write(format_exc())
    async def stop(self):
        self.peers.clear()
        self.queue_in.close()
        self.queue_out.close()
        self.subprocess.terminate()
        cuda.empty_cache()
        await super().stop()
        self.crash_callback(True)
        
        
    def put_on_queue(self,task):
        if self.has_weights:
            self.queue_out.put(task, True)
        else:
            self.deferred_tasks.append(task)
    @bindfrom("connected_callback")
    def peer_connected(self, nodeid, peer: Peer):
        
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"CONNECTED WITH {peer.pub_key}\n")
        msg = bytearray()
        msg += PPProtocl.INTRODUCTION.to_bytes(1,byteorder="big")
        msg += int(self.meshid).to_bytes(2,"big")
        msg += int(self.stage).to_bytes(2,"big")
        msg += self.peer.id_node
        msg += int(1).to_bytes(1,"big") if self.has_weights else int(0).to_bytes(1,"big")
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            log.write(f"INTRODUCING TO {peer.pub_key} {msg[0]} {p.addr}\n")
        loop = asyncio.get_event_loop()
        loop.create_task(self.send_datagram(msg, p.addr))

        return self.connected_callback(nodeid,peer)
        
 
    def process_datagram(self, addr: tuple[str, int], data: bytes):
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            log.write(f"datagram {data[0]} {addr}\n")
        
        if data[0] == PPProtocl.AGGREGATE_FLAG:
  
            self.received_aggregates += 1
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"AGGREGATE RECEIVED\n")
            if self.received_aggregates >= 3:
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"AGGREGATING\n")
                self.received_aggregates = 0
                self.processed.clear()
                self.send_receives.clear()
                self.deferred.clear()
                self.memory = self.MAX_MEM
                self.put_on_queue(Aggregate(0))
        elif data[0] == PPProtocl.INTRODUCTION:
            
            meshid = int.from_bytes(data[1:3],"big")
            stage = int.from_bytes(data[3:5],"big")
            nodeid = data[5:37]
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"INTRODUCTION FROM {meshid} {stage} {data[37]}\n")
            has_weights = data[37] == 1
            self.peers[meshid] = LocalPeer(self._lower_get_peer(nodeid),stage,meshid,has_weights)
            
            # Weight recovery
            if not self.has_weights and not self.requested and has_weights and stage == self.stage:
                self.requested = True
                msg = bytearray()
                msg += PPProtocl.MODEL_REQUEST_FLAG.to_bytes(1,byteorder="big")
                msg += self.peer.id_node
                loop = asyncio.get_event_loop()
                loop.create_task(self.send_datagram(msg, addr))

            if not self.has_weights and stage == self.stage and not has_weights:
                self.same_stage_without_weights += 1
            
            if not self.has_weights and self.same_stage_without_weights >= self.stage_size - 1:
                if self.stage == 1:
                    self.request_lr[0] = True
                elif self.stage == self.MAX_STAGE - 1:
                    self.request_lr[1] = True
                for p in self.peers.values():
                    if self.stage > 1 and p.stage == self.stage - 1 and not self.request_lr[0]:
                        self.request_lr[0] = True
                        msg = bytearray()
                        msg += PPProtocl.MODEL_REQUEST_FLAG.to_bytes(1,byteorder="big")
                        msg += self.peer.id_node
                        loop = asyncio.get_event_loop()
                        loop.create_task(self.send_datagram(msg, p.peer.addr))
                    elif self.stage < self.MAX_STAGE - 1 and p.stage == self.stage + 1 and not self.request_lr[1]:
                        self.request_lr[1] = True
                        msg = bytearray()
                        msg += PPProtocl.MODEL_REQUEST_FLAG.to_bytes(1,byteorder="big")
                        msg += self.peer.id_node
                        loop = asyncio.get_event_loop()
                        loop.create_task(self.send_datagram(msg, p.peer.addr))

        elif data[0] == PPProtocl.MODEL_REQUEST_FLAG:
            p = self._lower_get_peer(data[1:])
            
            self.put_on_queue(SendWeights(p.pub_key,None))

        
        return

   
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None
    
    
        
        
        

    @bindfrom("stream_callback")
    def process_data(self, data:bytes, nodeid, addr):

        if data[0] == PPProtocl.FORWARD_FLAG:
            
            bid = int.from_bytes(data[1:5],byteorder="big")
            frm = int.from_bytes(data[5:9],byteorder="big")
            self.send_receives[bid] = frm
            
            originator = int.from_bytes(data[9:13],byteorder="big")
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"Received from {frm} mb {bid} originator {originator} {time()}\n")
            
            
            
            nxt = None if self.stage == self.MAX_STAGE-1 else int(self.peers[self.meshid + self.stage_size].peer.pub_key)
            if nxt == None and self.peer.pub_key != str(originator):
                nxt = originator
            elif self.peer.pub_key == str(originator):
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"NEED TO COMPUTE LOSS FROM {frm} mb {bid}\n")
                
                self.put_on_queue(Loss(bid, frm, frm, originator, data[13:]))
                return

            
            
            
            self.put_on_queue(Forward(bid, frm, nxt, originator, data[13:]))

            return
        elif data[0] == PPProtocl.BACK_FLAG:
            
            bid = int.from_bytes(data[1:5],byteorder="big")
            frm = int.from_bytes(data[5:9],byteorder="big")
            originator = int.from_bytes(data[9:13],byteorder="big")
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"Will receive backward from {frm} mb {bid} originator {originator} {time()}\n")
           
            
            nxt = self.send_receives.get(bid)

            if originator == self.peer.pub_key:
                nxt = -1
                
            del self.send_receives[bid]
            self.put_on_queue(Backward(bid, frm, nxt, originator, data[13:]))
        elif data[0] == PPProtocl.MODEL_RESPONSE_FLAG:
            if self.has_weights:
                return
            if self.same_stage_without_weights >= self.stage_size - 1 and self.stage != 1 and self.stage != self.MAX_STAGE - 1:
                if self.store_other == None:
                    self.store_other = data[1:]
                else:
                    self.has_weights = True
                    self.put_on_queue(Weights(0,data[1:],self.store_other))
                    for v in self.deferred_tasks:
                        self.put_on_queue(v)
                    self.deferred_tasks.clear()
            else:
                self.has_weights = True
                self.put_on_queue(Weights(0,data[1:],None))
                for v in self.deferred_tasks:
                    self.put_on_queue(v)
                self.deferred_tasks.clear()
        elif data[0] == PPProtocl.GRADIENTS_FLAG:
           
            self.put_on_queue(Gradients(0,data[1:]))
            

        
       
    async def send_stream(self, node_id, data):
        
        await self._lower_find_peer(bytes(node_id))
        p = self._lower_get_peer(node_id)
        await self._lower_open_connection(p.addr[0], p.tcp, p.id_node, port_listen = 0)
        
        await self._lower_send_stream(node_id, data)
        return
    
    @bindto("open_connection")
    async def _lower_open_connection(self, remote_ip, remote_port, node_id: bytes):
        return
    @bindto("send_stream")
    async def _lower_send_stream(self, node_id, data):
        return
                
    def get_lowest_stream(self):
        submodule = self.submodule
        while submodule != None and not hasattr(submodule, "get_lowest_stream") and hasattr(submodule, "submodule") :
            submodule = submodule.submodule
        if submodule != None and hasattr(submodule, "get_lowest_stream"):
            ret = submodule.get_lowest_stream()
            if ret == None:
                return self
            else:
                return ret
        else:
            
            return self