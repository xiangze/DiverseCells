import __future__
import typing
import numpy as np
import numpy.random as npr
import copy
from dataclasses import dataclass
import random
import sys

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
    reactions:typing.List[Reaction]

def calcEP(cell:Cell,reactions:typing.List[Reaction]):
    ep=0
    for r in reactions:#反応種ごと
        ps=r.ps(cell)
        pt=r.pt(cell)
        ep=ep+(ps-pt)*np.log(ps/pt)
        ep=ep+(pt-ps)*np.log(pt/ps)
    return ep

def totalEP(cells:list,reactions:typing.List[Reaction]):
    return np.sum([calcEP(c,reactions) for c in cells])

def run(cells:list,reactions:list,Ds:list,externalchemicals,dt=0.01):
    ncells=cells.copy()
    N=len(cells)
    for i,c in enumerate(cells):
        #reactions
        for p in range(len(c.chemicals)):#chemical index
            ncells[i].chemicals[p]=c.chemicals[p]
            #chemical reactions
            for r in reactions:
                if(p in r.source):
                    ncells[i].chemicals[p]=c.chemicals[p]-r.pt(c)*dt
                elif(p in r.target):
                    ncells[i].chemicals[p]=c.chemicals[p]+r.ps(c)*dt
            #diffusions            
            ncells[i].chemicals[p]+=Ds[p]["inter"]*(cells[(i-1+N)%N].chemicals[p]+cells[(i+1)%N].chemicals[p]-2*c.chemicals[p])*dt
            ncells[i].chemicals[p]+=Ds[p]["global"]*(externalchemicals[p] -c.chemicals[p])*dt
    cells=ncells                    

sample=npr.random_sample
randint=npr.randint

class Cells():
    """
    Nc num. of cells
    M  num. of chemical spieces
    Nr num. of reactions
    """
    def __init__(self,Nc,M,Nr):
        self.reactions=[ Reaction([randint(M)],[randint(M)],[randint(M)]) for i in range(Nr)] #index
        self.cells=[Cell([sample() for _ in range(M) ],self.reactions) for _ in range(Nc)]
        self.Ds=[{"global":sample(), "inter":sample()} for _ in range(M)]
        
    def calcEP(self):
        return [calcEP(c,self.reactions) for c in self.cells]
    def population(self):
        return [c.chemicals for c in self.cells]
    
    def dump(self,fp=sys.stdout):
        print(self.population(),file=fp)

    def run(self,externalchemicals,dt=0.05):
        run(self.cells,self.reactions,self.Ds,externalchemicals,dt)

#前処理
if __name__=="__main__":
    dt=0.05        
    T=10000
    Nc=100
    M=30
    Nr=20

    cells=Cells(Nc,M,Nr)
    externalchemicals=[sample() for _ in range(M)]
    EP=[]
    history=[]
    for t in range(T):
        cells.run(externalchemicals,dt)
        if(t%100==0):
            eps=cells.calcEP()
            EP.append(eps)
            history.append(cells.population())
            
    EP=np.array(eps)
    np.savetxt("EntropyProd.csv",EP)
    history=np.array(history)
    np.savetxt("history.csv",history)
