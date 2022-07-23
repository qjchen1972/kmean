"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import math
import logging
from tqdm import tqdm
import numpy as np
import os
import torch
from torch.utils.data.dataloader import DataLoader


logger = logging.getLogger(__name__)


class KmeanConfig:
    dead_cluster = 10
    max_epochs = 10
    batch_size = 64
    ncluster = 512
    last_model = None
    model_path = 'model/'
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            

class KmeanTrainer:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.cfg = config        
        
        #cpu模式更快
        self.device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        if config.last_model is None:
            self.cluster = self.initCluster()
            self.start_epoch = 0
        else:
            self.cluster, self.start_epoch, loss =\
            self.load_model(os.path.join(self.cfg.model_path, config.last_model))
    
    def initCluster(self):    
            
        loader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                            batch_size=self.cfg.ncluster,
                            num_workers=0)
        
        return next(iter(loader))        
    
    def save_model(self, epoch, loss, cluster):
        torch.save({'epoch':epoch, 
                    'loss':loss, 
                    'cluster':cluster},
                    os.path.join(self.cfg.model_path, 'cluser_%d.pt' %(epoch + 1)),
                   )                    
    
    @classmethod    
    def load_model(cls, path):
        state_dict = torch.load(path)
        return state_dict['cluster'], state_dict['epoch']+1, state_dict['loss']
        
    @classmethod    
    def data2cluster(cls, x, c):
        if isinstance(x, list):
            x = torch.from_numpy(np.array(x))
        elif isinstance(x, np.ndarray):    
            x = torch.from_numpy(x)      
        return cls.data2cluster_euc(x, c) 
    
    
    
    '''
    Euclidean distance
    input：
      x: 输入向量 
      c：cluster
    return：
      x: 返回输入向量，可能会被单位化
      idx: x对应cluster的索引索引
      loss: x - c 的距离累计             
    '''
    @classmethod
    def data2cluster_euc(cls, x, c):
        temp = x[:, None, :] - c[None, :, :]
        temp = temp ** 2
        temp = temp.sum(-1)
        value, idx = temp.min(dim=1)
        return x, idx, value.sum()
    
    '''
    cosine distance
    input：
      x: 输入向量 
      c：cluster
    return：
      x: 返回输入向量，可能会被单位化
      idx: x对应cluster的索引索引
      loss: x和c的cos距离和            
    '''    
    @classmethod    
    def data2cluster_cos(cls, x, c):
        #转为单位向量
        x = x / torch.clamp(torch.norm(x, 2, dim=1)[:, None], 
                            min=torch.finfo(x.dtype).eps)   
        c = c / torch.clamp(torch.norm(c, 2, dim=1)[:, None], 
                            min=torch.finfo(c.dtype).eps)
        #找出cos值最大的                    
        temp = x[:, None, :] * (c[None, :, :])                
        temp = temp.sum(-1)
        value, idx = temp.max(dim=1)
        return x, idx, value.sum()
    
    
    def train(self):
        
        device = self.device        
        cluster = self.cluster.to(device)
        
        def run_epoch(c):
            device = self.device
        
            loader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                            batch_size=self.cfg.batch_size,
                            num_workers=0)
                            
            pbar = tqdm(enumerate(loader), total=len(loader))       
            total_loss = []
            
            new_cluster = torch.zeros(c.shape).to(device)
            new_cluster_num = torch.zeros(c.shape[0]).to(device)
            
            for it, x in pbar:  
                x = x.to(device)
                with torch.set_grad_enabled(False):
                    #若是cos距离，需要把x转为单位向量，来求出新的cluster
                    x, idx, loss,  = self.data2cluster(x, c)
                    total_loss.append(loss)
                    for k in range(self.cfg.ncluster):
                        t = x[idx==k]
                        new_cluster_num[k] += t.shape[0]
                        new_cluster[k] += t.sum(dim=0)
                        
                pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}")
            logger.info('train loss = %.3f' %(sum(total_loss))) 
            new_cluster =  new_cluster / new_cluster_num[:, None]        
            return new_cluster, new_cluster_num, total_loss    
            
        for epoch in range(self.start_epoch, self.cfg.max_epochs):      
            newc, newc_num, loss = run_epoch(cluster)
            #替换掉一些cluster,
            #nanix = torch.any(torch.isnan(newc), dim=1)
            #ndead = nanix.sum().item()
            nanix = newc_num < self.cfg.dead_cluster
            ndead = nanix.sum()            
            print('done step %d/%d, re-initialized %d dead clusters' % (epoch+1, self.cfg.max_epochs, ndead))
            if ndead > 0:
                clu =  self.initCluster().to(device)
                newc[nanix] = clu[torch.randperm(clu.shape[0])[:ndead]]
                del clu      
            cluster = newc
            self.save_model(epoch, sum(loss), cluster.cpu())
            