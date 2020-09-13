# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:32:00 2020

@author: Administrator
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('creditcard.csv')
RANDOM_SEED = 42

plt.subplot(211)
df.query('Class==1').Amount.plot(kind='hist',bins=50,xlim=(0,100),logy=True,grid=True,title='Fraud')
plt.subplot(212)
df.query('Class==0').Amount.plot(kind='hist',bins=50,xlim=(0,100),logy=True,grid=True,title='Normal')


## Time Matters??
plt.subplot(211)
df.query('Class==1').plot(kind='scatter',x='Time',y='Amount',grid=True,title='Fraud')
plt.subplot(212)
df.query('Class==0').plot(kind='scatter',x='Time',y='Amount',grid=True,title='Normal')



df.Amount = StandardScaler().fit_transform(df[['Amount']]).ravel()
Normal = df.loc[df.Class==0,'V1':'Amount'].values
Fraud = df.loc[df.Class==1,'V1':'Amount'].values


np.save('./Normal',Normal)
np.save('./Fraud',Fraud)



# =============================================================================
# RUN IN PYTORCH ENV
# =============================================================================
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.utils.data as Data
from visdom import Visdom



class CCFD(Data.Dataset):
    
    def __init__(self,path = r'./Normal.npy'):
        self.data = np.load(path)
        self.data = t.from_numpy(self.data).float()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,index):
        return self.data[index]
        
Normal = CCFD('./Normal.npy')
train_db, val_db = Data.random_split(Normal, [227451, 56864])
Fraud= CCFD('./Fraud.npy')

class AutoEncoder(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(29,14),
            nn.Tanh(),
            nn.Linear(14,7),
            nn.ReLU(),
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(7,7),
            nn.Tanh(),
            nn.Linear(7,29),
            nn.ReLU()
            )

    def forward(self,x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()
optimizer = t.optim.Adam(autoencoder.parameters())
loss_func = nn.MSELoss()

train_loader = Data.DataLoader(dataset=train_db,batch_size=32,shuffle=True)


# Hyper Parameters
EPOCH = 100
BATCH_SIZE = 32


# 创建窗口并初始化
viz = Visdom() 
viz.line([[np.nan,np.nan]], [np.nan], win='train',
         opts=dict(title='errors', legend=['train', 'test']))

for epoch in range(EPOCH):
    
    train_error = 0
    for step, x in enumerate(train_loader):
        
        # 读取批数据
        b_x, b_y = x.view(*x.shape), x.view(*x.shape)
        
        # 计算模型输出
        encoded, decoded = autoencoder(b_x)
        
        # 计算模型输出的损失
        loss = loss_func(decoded, b_y)

        # 清除最后的误差梯度
        optimizer.zero_grad()
        
        # 通过模型反向传播错误
        loss.backward()
        train_error += loss.detach()
        
        # 更新模型参数
        optimizer.step()
        
    train_error = train_error.numpy()/len(train_loader)
    yhat = autoencoder(Normal[val_db.indices])[1]
    val_loss = loss_func(yhat,Normal[val_db.indices]).detach().numpy()
    
    viz.line([[train_error, val_loss]], [epoch], win='train', update='append')
    
    print(f'Epoch: {epoch}, | train loss: {train_error:.4f}, val loss: {val_loss:.4f}')


test = t.cat([Normal[val_db.indices],Fraud.data])
pred = autoencoder(test)[1]
rec_error = (pred-test).square().mean(1).detach().numpy()
label = np.repeat([0,1],[len(val_db.indices),len(Fraud)])
error_df = pd.DataFrame({'ERR':rec_error,'LAB':label})
plt.figure()
error_df.query('LAB==0 and ERR<10').ERR.hist(bins=50)
plt.figure()
error_df.query('LAB==1').ERR.hist(bins=500)


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, RocCurveDisplay, PrecisionRecallDisplay)
## roc
fpr, tpr, _ = roc_curve(error_df.LAB,error_df.ERR)
roc_auc = auc(fpr, tpr)
RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc).plot()

## pr
p, r, thres = precision_recall_curve(error_df.LAB, error_df.ERR)
PrecisionRecallDisplay(p,r).plot()
plt.figure()
plt.plot(thres,p[1:],marker='o')
plt.figure()
plt.plot(thres,r[1:],marker='o')

## confusion matrix
threshold = 2.9
error_df['PRED'] = error_df.ERR>threshold
confusion_matrix(error_df.LAB,error_df.PRED)
