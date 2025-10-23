import __future__
import typing
import numpy as np
import numpy.random as npr
import copy
from dataclasses import dataclass
import random
import sys
import matplotlib.pyplot as plt

@dataclass
class Reaction():
    source:typing.List[int]
    target:typing.List[int]
    enzymes:typing.List[int]
    def ps(self,c):#forward
        return self.enzymes[0]*np.prod([c.chemicals[i] for i in self.source])
    def pt(self,c):#backward
        return self.enzymes[1]*np.prod([c.chemicals[i] for i in self.target])

@dataclass    
class Reaction_nlin(Reaction):
    sigma:float=1.
    th:float=1.
    def ps(self,c):#forward
        return np.tanh(self.enzymes[0]*np.prod([c.chemicals[i] for i in self.source])-self.th)
    def pt(self,c):#backward
        return np.tanh(self.enzymes[1]*np.prod([c.chemicals[i] for i in self.target])-self.th)
        
@dataclass    
class Cell():
    chemicals:typing.List[float]
    #reactions:typing.List[Reaction]

SMALLVAL=1e-8
def calcEP(cell:Cell,reactions:typing.List[Reaction]):
    ep=0
    for r in reactions:#反応種ごと
        ps=r.ps(cell)+1e-8
        pt=r.pt(cell)+1e-8
        try:
            ep=ep+2*((ps-pt)*np.log(ps/pt))
        except:
            ep=1e+10
    return ep

def totalEP(cells:list,reactions:typing.List[Reaction]):
    return np.sum([calcEP(c,reactions) for c in cells])

def run(cells:list,reactions:list,Ds:list,externalchemicals,dt=0.01,debug=False):
    ncells=copy.deepcopy(cells)
    N=len(cells)
    for i,c in enumerate(cells):
        nc=ncells[i]
        #reactions
        if(debug):
            print(f"{i}th cell")
        for p in range(len(c.chemicals)):#chemical index
            nc.chemicals[p]=c.chemicals[p]
            #chemical reactions
            for r in reactions:
                if(p in r.source):
                    nc.chemicals[p]=c.chemicals[p]-r.pt(c)*dt
                elif(p in r.target):
                    nc.chemicals[p]=c.chemicals[p]+r.ps(c)*dt
            #diffusions            
            nc.chemicals[p]+=Ds[p]["inter"]*(cells[(i-1+N)%N].chemicals[p]+cells[(i+1)%N].chemicals[p]-2*c.chemicals[p])*dt
            #dilutons?
            nc.chemicals[p]+=Ds[p]["global"]*(externalchemicals[p] -c.chemicals[p])*dt
    return ncells

def sample(r=1e-8):
    return npr.random_sample()+r

randint=npr.randint

class Cells():
    """
    Nc num. of cells
    M  num. of chemical spieces
    Nr num. of reactions
    """
    def __init__(self,Nc=100,M=30,Nr=20,dilute="gradient"):
        self.reactions=[ Reaction([randint(M-1)],[randint(M-1)],[randint(M-1),randint(M-1)]) for i in range(Nr)] #index
        self.cells=[Cell([sample() for _ in range(M) ]) for _ in range(Nc)]
        self.totsize=Nc*M    
        if(dilute=="gradient"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample((M-i)*0.1), "inter":sample((M-i)*0.1)} for i in range(M)]
        elif(dilute=="gradient_exp"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample(2**((M-i)/M)), "inter":sample(2**((M-i)/M))} for i in range(M)]
        else:#random
            self.Ds=[{"global":sample(), "inter":sample()} for _ in range(M)]


    def calcEP(self):
        return [calcEP(c,self.reactions) for c in self.cells]

    def calcEntrotpy(self):#細胞間多様性
        return np.sum([p*np.log(p) for c in self.cells for p in c.chemicals ])
    
    def population(self):
        return [c.chemicals for c in self.cells]
    
    def dump(self,fp=sys.stdout):
        print(self.population(),file=fp)

    def run(self,externalchemicals,dt=0.05):
        self.cells=run(self.cells,self.reactions,self.Ds,externalchemicals,dt)

    def run_all(self,T,externalchemicals,dt=0.05,suffix="",peri=100,debug=False,plot=False):        
        EP=[]
        history=[]
        for t in range(T):
            if(debug):
                print(t)
            self.run(externalchemicals,dt)
            if(t%peri==0):
                eps=self.calcEP()
                EP.append(eps)
                history.append(self.population())
                
        EP=np.array(eps)
        np.savetxt(f"EntropyProd_{suffix}.csv",EP)
        history=np.array(history).reshape(self.totsize,T//peri)
        np.savetxt(f"history_{suffix}.csv",history)
        if(plot):
            plt.plot(history)
            plt.savefig(f"history_{suffix}.png")            
            plt.clf()
            plt.close()
            plt.plot(EP)
            plt.savefig(f"EP_{suffix}.png")            

class SignalCells(Cells):
    def __init__(self,Nc,M,Nr):
        super().__init__(Nc,M,Nr)
        Nr=M*2
        #pairindex
        self.reactions=[ Reaction([i],[i+1],[max(i+2,M),min(i-2,0)]) for i in range(Nr-1)] 

#前処理
if __name__=="__main__":
    dt=0.01        
#    T=10000
    T=1000
    Nc,M,Nr=100,30,20
    cells=Cells(Nc,M,Nr)
    externalchemicals=[sample() for _ in range(M)]
    cells.run_all(T,externalchemicals,dt,"today",plot=True)