# -*- coding:utf-8 -*-
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
import os
from trainer import KmeanTrainer, KmeanConfig

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,        
        #filename= 'dat.log',
)
logger = logging.getLogger(__name__)

tr_path = 'data/te.npy'

class StockDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        #数据格式是 n*1*6
        self.data = np.load(path,allow_pickle=True)
        print(self.data.shape)
      
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        # 读出来的数据是strt类型, 需要转换
        return torch.from_numpy(self.data[idx,0,2:].astype(np.float32))
    
    def getinfo(self, idx):
        return self.data[idx,0,:2]    

train_dataset = StockDataset(tr_path)
print(len(train_dataset))
print(train_dataset.getinfo(0))
print(train_dataset[0])
logger.info('ok')


kconf = KmeanConfig(max_epochs=10, 
                    batch_size=512, 
                    ncluster=2048, 
                    dead_cluster=3,
                    #last_model='cluser_%d.pt' %(43),
                    )

trainer = KmeanTrainer(train_dataset, kconf)
trainer.train()


cluster, cluster_num, epoch, loss = KmeanTrainer.load_model('model/cluser_5.pt')
print(cluster, cluster_num, epoch, loss)
v, m = cluster_num.sort(descending=True)
print(v, m)
print(cluster[m[0]], cluster[m[1]], cluster[m[2]])
#cluster = trainer.cluster

x = [[0.032, 0.049, -0.025, 0.015]]
idx, loss = KmeanTrainer.data2cluster(x, cluster)
print(idx, loss, cluster[idx])