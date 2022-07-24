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
    top_select = 10
    select_threahold = 0.2
    
    dead_cluster = 20
    max_epochs = 10
    batch_size = 64
    ncluster = 512
    last_model = None
    model_path = 'model/'
    dis_mode = 'euc'
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
top_select = 10
select_threahold = 1.2
    
class KmeanTrainer:

    
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.cfg = config        
        
        #cpu模式更快
        self.device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        if config.last_model is None:
            self.cluster = self.initCluster()
            self.cluster_num = torch.zeros(self.cluster.shape[0])
            self.start_epoch = 0
        else:
            self.cluster, self.cluster_num, self.start_epoch, loss =\
            self.load_model(os.path.join(self.cfg.model_path, config.last_model))
    
    def initCluster(self):    
            
        loader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                            batch_size=self.cfg.ncluster,
                            num_workers=0)
        
        return next(iter(loader))        
    
    def save_model(self, epoch, loss, cluster, cluster_num):
        torch.save({'epoch':epoch, 
                    'loss':loss, 
                    'cluster':cluster,
                    'cluster_num':cluster_num},
                    os.path.join(self.cfg.model_path, 'cluser_%d.pt' %(epoch + 1)),
                   )                    
    
    @classmethod    
    def load_model(cls, path):
        state_dict = torch.load(path)
        return state_dict['cluster'], state_dict['cluster_num'],\
               state_dict['epoch']+1, state_dict['loss']
    
    
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
        return idx, value.sum()
            
     
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
        return idx, value.sum()
    
    
    def dataproc(self, x, c, c_num=None):
        if self.cfg.dis_mode == 'euc':
            return self.dataproc_euc(x, c, c_num)
        elif self.cfg.dis_mode == 'cos':
            return self.dataproc_cos(x, c, c_num)        
        
    def _selectidx(self, data, c_num, large=False):
        if large is True:
            value, idx = data.max(dim=1)
        else:
            value, idx = data.min(dim=1)
        return  idx, value           
        #选择最可能的10个值, 从这些值里选满足阈值的值，然后优先分配给数目最大的cluster
        value, idx = torch.topk(data, 
                                k=self.cfg.top_select, 
                                dim=-1, 
                                largest=large)
                
        if large is True:
            value = torch.flip(value, dims=[-1])
            idx = torch.flip(idx, dims=[-1])
        
        ans_idx = []
        ans_value = []
        for it, (i, v) in enumerate(zip(idx, value)):
            if large is True:
                sel = v >= v[0] * (1.0 - self.cfg.select_threahold)
            else:            
                sel = v <= v[0] * (1.0 + self.cfg.select_threahold)
            mv, mi = c_num[i[sel]].max(dim=-1)     
            ans_idx.append(i[sel][mi])   
            ans_value.append(v[sel][mi])
            
        ans_idx = torch.tensor(ans_idx, dtype=torch.int)
        ans_value = torch.tensor(ans_value, dtype=torch.float32)
        return ans_idx, ans_value
        
    
    def dataproc_euc(self, x, c, c_num):
    
        temp = x[:, None, :] - c[None, :, :]
        temp = temp ** 2
        temp = temp.sum(-1)
        ans_idx, ans_value = self._selectidx(temp, c_num, large=False)
        
        return x, ans_idx, ans_value.sum()
    
    def dataproc_cos(self, x, c, c_num):
        #转为单位向量
        x = x / torch.clamp(torch.norm(x, 2, dim=1)[:, None], 
                            min=torch.finfo(x.dtype).eps)   
        c = c / torch.clamp(torch.norm(c, 2, dim=1)[:, None], 
                            min=torch.finfo(c.dtype).eps)
        #找出cos值最大的                    
        temp = x[:, None, :] * (c[None, :, :])                
        temp = temp.sum(-1)
        ans_idx, ans_value = self._selectidx(temp, c_num, large=True)        
        return x, ans_idx, ans_value.sum()
    
    
    def train(self):
        
        device = self.device        
        cluster = self.cluster.to(device)
        cluster_num = self.cluster_num.to(device) #torch.zeros(cluster.shape[0]).to(device)
        #cluster_num = torch.randint(0, 100, (cluster.shape[0],))
        def run_epoch(c, c_num):
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
                    x, idx, loss,  = self.dataproc(x, c, c_num)
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
            newc, newc_num, loss = run_epoch(cluster, cluster_num)
            self.save_model(epoch, sum(loss), newc.cpu(), newc_num.cpu())
            #替换掉一些cluster,
            #nanix = torch.any(torch.isnan(newc), dim=1)
            #ndead = nanix.sum().item()
            nanix = newc_num < self.cfg.dead_cluster
            ndead = nanix.sum()            
            print('done step %d/%d, re-initialized %d dead clusters' % (epoch+1, self.cfg.max_epochs, ndead))
            if ndead > 0:
                clu =  self.initCluster().to(device)
                newc[nanix] = clu[torch.randperm(clu.shape[0])[:ndead]]
                newc_num[nanix] = torch.zeros(ndead).to(device)
                del clu
                
            cluster = newc
            cluster_num = newc_num
           
            