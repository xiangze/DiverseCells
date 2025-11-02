import __future__
import typing
import numpy as np
import numpy.random as npr
import copy
from dataclasses import dataclass
import random
import sys
import matplotlib.pyplot as plt
import datetime

@dataclass
class Reaction():
    source:typing.List[int]
    target:typing.List[int]
    enzymes:typing.List[int]
    def ps(self,c)->float:#forward
        return self.enzymes[0]*np.prod([c.chemicals[i] for i in self.source])
    def pt(self,c)->float:#backward
        return self.enzymes[1]*np.prod([c.chemicals[i] for i in self.target])

@dataclass    
class Reaction_nlin(Reaction):
    sigma:float=1.
    th:float=1.
    def ps(self,cell):#forward
        return np.tanh(self.enzymes[0]*np.prod([cell.chemicals[i] for i in self.source])-self.th)
    def pt(self,cell):#backward
        return np.tanh(self.enzymes[1]*np.prod([cell.chemicals[i] for i in self.target])-self.th)
        
@dataclass    
class Cell():
    chemicals:typing.List[float]
    def calcVolume(self):
        return sum([p for p in self.chemicals[p] ])
    
calcep= lambda ps,pt:(ps-pt)*np.log(ps/pt)

SMALLVAL=1e-8
LARGEVAL=1e+10
def calcReactionEP(cell:Cell,reactions:typing.List[Reaction])->float:
    ep=0
    for r in reactions:#反応種ごと
        ps=r.ps(cell)+SMALLVAL
        pt=r.pt(cell)+SMALLVAL
        try:
            ep+=ep+2*calcep(ps,pt)#ep+2*((ps-pt)*np.log(ps/pt))
        except:
            ep+=LARGEVAL
    return ep


sample=lambda r=1e-8:npr.random_sample()+r
randint=npr.randint

class Cells():
    """
    Nc num. of cells
    M  num. of chemical spieces
    Nr num. of reactions
    """
    def __init__(self,Nc,M,Nr,externalchemicals,dilute="gradient",reactiontype="random",seed=42,growth=False,debug=False):
        npr.seed(seed)
        self.Nc=Nc
        self.M=M
        self.Nr=Nr
        self.growth=growth
        self.cells=[Cell([sample() for _ in range(M) ]) for _ in range(Nc)]
        self.totsize=Nc*M
        self.externalchemicals=externalchemicals

        if(reactiontype=="random"):
            self.reactions=[ Reaction([randint(M-1)],[randint(M-1)],[randint(M-1),randint(M-1)]) for i in range(Nr)] #index
        elif(reactiontype=="lattice"):
            self.reactions=[ Reaction([i],[i+1],[min(i,M),max(i+1,0)]) for i in range(Nr)] #index
        elif(reactiontype=="cascade" or reactiontype=="pair"):
            self.reactions=[ Reaction([i],[i+1],[min(i+2,M),max(i-2,0)]) for i in range(0,Nr-1,2)] 
        elif(reactiontype=="pararell"): #pararell
            num=5
            Nrr=Nr-1
            Nr=Nr*num
            self.reactions=[ Reaction([i+n*Nrr],[(n*N)*i+1],[max(n*N+i+2,N*M),min(n*N+i-2,0)]) for i in range(Nrr) for n in range(num)] 
        elif(reactiontype=="forward"): #like Resnet
            self.reactions=[ Reaction([i],[i+1],[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)]             
            self.reactions+=[ Reaction([i],[min(i+5)],[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)] #feed            
        elif(reactiontype=="backward"): 
            self.reactions=[ Reaction([i],[i+1],[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)]             
            self.reactions+=[ Reaction([i],[i+1],[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)]  #feed                       
        else:
            print("reaction mode:random,lattice,cascade,pararell,forward,backward feedback")

        if(dilute=="gradient"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample((M-i)*0.1), "inter":sample((M-i)*0.1)} for i in range(M)]
        elif(dilute=="reverse"):# 奥(index大)のほうが小さい            
            self.Ds=[{"global":sample((i+1)*0.1), "inter":sample((i+1)*0.1)} for i in range(M)]
        elif(dilute=="gradient_exp"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample(2**((M-i)/M)), "inter":sample(2**((M-i)/M))} for i in range(M)]
        elif(dilute=="reverse_exp"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample(2**(i/M)), "inter":sample(2**(i/M))} for i in range(M)]
        elif(dilute=="random"):
            self.Ds=[{"global":sample(), "inter":sample()} for _ in range(M)]
        elif(dilute=="constant"):
            self.Ds=[{"global":1, "inter":1.5} for _ in range(M)]
        else:
            print("diluut mode are[gradient,gradient_exp,reverse,random,constant]")

    def calcVolume(self,c:Cell)->float:
        return np.sum(c.chemicals)
    
    def calcTotalVolume(self)->float:
        return np.sum([self.calcVolume(c)] for c in self.cells )
   
    def calcReactionEP(self,ci:int)->float:
        return calcReactionEP(self.cells[ci],self.reactions) 

    def calcTotalReactionEP(self):
        return np.array([calcReactionEP(c,self.reactions) for c in self.cells]).sum()

    def calcDiffusionEP(self):
        EP=0
        N=self.Nc
        for i,c in enumerate(self.cells):#reactions
            for p in range(self.M):#chemi
                #diffusions            
                EP+=self.Ds[p]["inter"]*(self.cells[(i-1+N)%N].chemicals[p]-c.chemicals[p])
                EP+=self.Ds[p]["inter"]*(c.chemicals[p]-self.cells[(i+1)%N].chemicals[p])
                #dilutons?
                EP+=self.Ds[p]["global"]*calcep(self.externalchemicals[p] ,c.chemicals[p])
        return EP
    
    def calcEP(self,ci:int):
        return self.calcReactionEP(ci)+self.calcDiffusionEP()
    def calcTotalEP(self):
        return sum([self.calcEP(ci) for ci in range(self.Nc) ])

    def calcEntropy(self):#細胞間多様性
        return np.sum([p*np.log(p) for c in self.cells for p in c.chemicals ])
    #
    def calcTotalEntropy(self,dt):
        E0=self.calcEntropy()
        self.run(dt)
        EP=self.calcTotalEP()
        E1=self.calcEntropy()
        return ((E1-E0),EP*dt)

    def calcEntropy_dif(self,dt):
        Ed,EP=self.calcTotalEntropy(dt)
        return EP-Ed
    
    def population(self):
        return [c.chemicals for c in self.cells]
    
    def dump(self,fp=sys.stdout):
        print(self.population(),file=fp)

    def run(self,dt:float=0.,debug=False):
        ncells=copy.deepcopy(self.cells)
        N=self.Nc
        for i,c in enumerate(self.cells):
            nc=ncells[i]
            #reactions
            if(debug):
                print(f"{i}th cell")
            for p in range(len(c.chemicals)):#chemical index
                nc.chemicals[p]=c.chemicals[p]                
                for r in self.reactions: #chemical reactions
                    if(p in r.source):
                        print(r.pt(c),dt)
                        nc.chemicals[p]=c.chemicals[p]-r.pt(c)*dt
                    elif(p in r.target):
                        nc.chemicals[p]=c.chemicals[p]+r.ps(c)*dt
                #diffusions            
                nc.chemicals[p]+=self.Ds[p]["inter"]*(self.cells[(i-1+N)%N].chemicals[p]+self.cells[(i+1)%N].chemicals[p]-2*c.chemicals[p])*dt
                #dilutons?
                nc.chemicals[p]+=self.Ds[p]["global"]*(self.externalchemicals[p] -c.chemicals[p])*dt
            if(self.growth):
                gamma=self.calcVolume(c)
                for p in range(len(c.chemicals)):
                    nc.chemicals[p]-=gamma*c.chemicals[p]*dt
        self.cells=ncells

    def divide(self):
        for i,c in enumerate(self.cells):
            if(self.calcVolume(c)>self.maxV):
                ncell=copy.deep(self.c)
                for p in range(len(c.chemicals)):
                    ncell.chemicals[p]+=sample()*1e-7
                self.cells.insert(i,ncell)
                self.Nc=Nc+1

    def run_all(self,T,dt=0.05,suffix="",peri=100,debug=False,plot=True):
        self.EP=[]
        history=[]
        totEnt=[]
        for t in range(T):
            if(debug):
                print(t)
            self.run(dt)
            if(t%peri==0):
                eps=self.calcTotalEP()
                self.EP.append(eps)
                history.append(self.population())
                totEnt.append(self.calcEntropy_dif(dt))

                if(self.growth):
                    self.divide()

        self.totEnt=np.array(totEnt)
        self.EP=np.array(eps)
        self.history=np.array(history).reshape(self.totsize,T//peri)
        self.save(suffix)
        if(plot):        
            self.plots(suffix)
            for p in range(self.M):
                celldist=[ c.chemicals[p] for c in enumerate(self.cells)]
                plt.plot(celldist)
            plt.savefig(f"last_cell_distribusion_{suffix}.png")            
            plt.clf()
            plt.close()

    def plots(self,suffix):
        for k,v in {"history":self.history,"EP":self.EP,"totalEntropy":self.totEnt}.items():
            plt.plot(v)
            plt.savefig(f"{k}_{suffix}.png")            
            plt.clf()
            plt.close()

    def save(self,suffix=""):
        np.savetxt(f"EntropyProd_{suffix}.csv",self.EP)
        np.savetxt(f"totalEntropy_{suffix}.csv",self.totEnt)
        np.savetxt(f"history_{suffix}.csv",self.history)

def run_default(T=1000,dt=0.01,Nc=200,r="random",d="gradient"):
    seed=0
    for M in [30,50,100,500]:
        for nr in [0.1,0.5,0.8]:
            now=datetime.datetime.now()
            today=f"{now.year}:{now.month}:{now.date}_{now.hour}:{now.minute}:{now.second}"
            Nr=int(M*nr)
            externalchemicals=[sample(seed) for _ in range(M)]
            cells=Cells(Nc,M,Nr,externalchemicals,dilute=d,reactiontype=r,seed=seed)
            cells.run_all(T,dt,f"Cells{Nc}_Ch{M}_r{nr}_{r}_{d}_seed{seed}_{today}")
            seed+=1

def run_allconds(T=1000,dt=0.01,Nc=200,
                 rtype=["random","lattice","cascade","pararell","forward","backward" "feedback"],
                 diftype=["gradient","gradient_exp","reverse","reverse_exp","random","constant"] ):
    seed=0
    for r in rtype:
        for d in diftype:
            for M in [30,50,100,500]:
                for nr in [0.1,0.5,0.8]:
                    now=datetime.datetime.now()
                    today=f"{now.year}:{now.month}:{now.date}_{now.hour}:{now.minute}:{now.second}"
                    Nr=int(M*nr)
                    externalchemicals=[sample(seed) for _ in range(M)]
                    cells=Cells(Nc,M,Nr,externalchemicals,dilute=d,reactiontype=r,seed=seed)
                    cells.run_all(T,dt,f"Cells{Nc}_Ch{M}_r{nr}_{r}_{d}_seed{seed}_{today}")
                    seed+=1

if __name__=="__main__":
    dt=0.01        
    T=1000
    Nc=200
    run_allconds(T=1000,dt=0.01,Nc=200,rtype=["random","cascade"],diftype=["gradient"])
