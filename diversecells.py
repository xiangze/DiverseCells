import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DiverseCells():
    def __init__(self,N,M, sparserate=0.7,seed=1,eps=0.001,dt=0.01):
        np.random.seed(seed)
        self.N=N #cell num
        self.M=M #chemical num
        self.k=np.maximum(0,np.random.random([M,M,M])-sparserate)#chemical coefficient 
        self.alpha=np.power(np.random.random(M),2) #nonlinear power
        self.D=np.power(np.random.random(M),2) #diffusion rate
        self.m=np.power(np.random.random(M),2) #growth death rate
        self.eps=eps #noise strength
        self.p=np.maximum(0, np.random.random([N,M])) #population
        self.dt=dt
        
    def initp(self,seed=1):
        np.random.seed(seed)
        self.p=np.maximum(0, np.random.random([N,M])) #population
        
    def showparam(self):
        print(self.N,self.M)
        print("chemical coefficient k")
        print(self.k)
        print("chemical coefficient alpha")
        print(self.alpha)
        print("decay rate m")
        print(self.m)
        print("diffusion rate D")
        print(self.D)
        print("noise strength eps")        
        print(self.eps)
        
    def step(self,dt):
        p_ave=np.mean(self.p,axis=0)
        palpha=np.power(self.p,self.alpha)
        pp=np.einsum("nil,nl->ni",np.einsum("jil,nj->nil", self.k, self.p),palpha)
        pn=np.einsum("nil,nl->ni",np.einsum("jil,nj->nil", self.k, self.p),palpha)       
        pd=self.D*(np.repeat(p_ave,N,axis=0).reshape([N,M])-self.p)
        pm=self.m*self.p
        return self.dt*(pp-pn-pm+pd+self.eps*np.random.poisson(self.p))
        
    def prop(self,dt):           
        self.p=self.p+self.step(dt)
        return self.p
    
    def timeseries(self,t):
        population=[self.prop(self.dt) for t in range(t)]
        return np.array(population)
           
#enviroment change    
    def with_shock(self,t,newD):
        population= self.timeseries(t/2)
        self.D=newD
        population2=self.timeseries(t/2)
        return np.array([population,population2])
    
    def calc_entropy(p):
        pp=p/np.mean(p)
        return -np.sum(pp*np.log(pp))    
    
class DiverseCells_gpu(torch.nn.Module):
    def __init__(self,N,M, sparserate=0.7,seed=1,eps=0.001):
        torch.manual_seed(seed)
        self.N=N #cell num
        self.M=M #chemical num
        self.k=torch.max(0,np.random.random([M,M,M])-sparserate)#chemical coefficient 
        self.alpha=torch.pow(np.random.random(M),2) #nonlinear power
        self.D=torch.pow(np.random.random(M),2) #diffusion rate
        self.m=torch.pow(np.random.random(M),2) #growth death rate
        self.eps=eps #noise strength
        self.p=np.max(0, np.random.random([N,M])) #population
        
    def initp(self,seed=1):
        torch.manual_seed(seed)
        self.p=torch.max(0, torch.rand((N,M))) #population
        return self.p
    
    def showparam(self):
        print(self.N,self.M)
        print("chemical coefficient k")
        print(self.k)
        print("chemical coefficient alpha")
        print(self.alpha)
        print("decay rate m")
        print(self.m)
        print("diffusion rate D")
        print(self.D)
        print("noise strength eps")        
        print(self.eps)

    def step(self,dt):
        p_ave=torch.mean(self.p,axis=0)
        palpha=torch.pow(self.p,self.alpha)
        pp=torch.einsum("nil,nl->ni",torch.einsum("jil,nj->nil", self.k, self.p),palpha)
        pn=torch.einsum("nil,nl->ni",torch.einsum("jil,nj->nil", self.k, self.p),palpha)       
        pd=self.D*(torch.reshape(torch.repeat(p_ave,self.N,axis=0),(self.N,self.M))-self.p)
        pm=self.m*self.p
        return self.dt*(pp-pn-pm+pd+self.eps*torch.poisson(torch.rand(self.N,self.M)*self.p))
    
    def prop(self,dt):           
        return self.p+self.step(dt)
        
    def timeseries(self,t):
        population=[self.prop(self.dt) for _ in range(t)]
        return np.array(population)
           
#enviroment change    
    def with_shock(self,t,newD):
        population= self.timeseries(t/2)
        self.D=newD
        population2=self.timeseries(t/2)
        return np.array([population,population2])

    def calc_entropy(p):
        pp=p/torch.mean(p)
        return -torch.sum(pp*torch.log(pp))
    
    def eveolve(self,nn,criterion,optimizer,period=1000,epochs=3):
        nn.to("cuda:0")
        nn.train()
        for i in range(epochs):
                x=self.initp
                for i in range(period):
                    x=nn(x)

                if(criterion=="grouth"):
                    loss=-self.step(x)
                else:
                    loss=-self.calc_entropy(x)

                loss.backward()
                optimizer.step()
                      
                # print statistics
                running_loss += loss.item()
                print(f'epoch[{i + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} ')
                running_loss = 0.0

        print('Finished Evolution path')

class DiverseCellsNet(nn.Module):
    def __init__(self,cells):
        super().__init__()
        self.cells=cells
        self.pror =cells.prop

#        nn.init.zeros_(self.cells.weight) # 重みの初期値を設定
#        nn.init.ones_(self.conv1.bias)    # バイアスの初期値を設定
    def forward(self, x):
        return self.prop(x)

if __name__=="__main__":
     cells=DiverseCells_gpu()
     net=DiverseCellsNet(cells)
     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
     cells.eveolve(nn,"grouth",optimizer)