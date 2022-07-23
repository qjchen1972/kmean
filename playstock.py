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

te_path = 'data/te.npy' 

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

tr_path = 'data/te.npy'        
train_dataset = StockDataset(tr_path)
print(len(train_dataset))
print(train_dataset.getinfo(0))
print(train_dataset[0])
logger.info('ok')


kconf = KmeanConfig(max_epochs=2, 
                    batch_size=512, 
                    ncluster=2048, 
                    dead_cluster=10,
                    #last_model='cluser_%d.pt' %(3),
                    )

trainer = KmeanTrainer(train_dataset, kconf)
trainer.train()


#cluster, epoch, loss = KmeanTrainer.load_model('model/cluser_30.pt')
#print(cluster, epoch, loss)
cluster = trainer.cluster

x = [[0.005, 0.03, -0.025, 0.015]]
_, idx, loss = KmeanTrainer.data2cluster(x, cluster)
print(idx, loss, cluster[idx])