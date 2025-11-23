import __future__
import typing
import numpy as np
import numpy.random as npr
import numpy.typing as npt
import copy
from dataclasses import dataclass
import random
import sys
import matplotlib.pyplot as plt
import datetime
import os
import warnings

@dataclass    
class Cell():
    chemicals:npt.ArrayLike
    def calcVolume(self):
        return np.sum(self.chemicals)

@dataclass    
class Reaction():
    source:typing.List[int] #index
    target:typing.List[int] #index
    enzymes:typing.List[int] #index
    def __init__(self,s,t,enz):
        self.source=s
        self.target=t
        self.enzymes=enz
    def ps(self,c:Cell)->float:#forward
        return c.chemicals[self.enzymes[0]]*np.prod([c.chemicals[i] for i in self.source])
    def pt(self,c:Cell)->float:#backward
        return c.chemicals[self.enzymes[1]]*np.prod([c.chemicals[i] for i in self.target])

@dataclass    
class Reaction1():
    source:int #index
    target:int #index
    enzymes:typing.List[int] #index
    def __init__(self,s,t,enz):
        self.source=s
        self.target=t
        self.enzymes=enz
    def ps(self,c:Cell)->float:#forward
        return c.chemicals[self.enzymes[0]]*c.chemicals[self.source]
    def pt(self,c:Cell)->float:#backward
        return c.chemicals[self.enzymes[1]]*c.chemicals[self.target]
    
    #forward
    def ps_s(self,cells:npt.ArrayLike)->float: 
        return cells[self.enzymes[0]]*cells[self.source]
    #backward
    def pt_s(self,cells:npt.ArrayLike)->float: 
        return cells[self.enzymes[1]]*cells[self.target]
    
    def calcepr(self,cell):
        return calcepr(self.ps_s(cell),self.pt_s(cell))
        
@dataclass    
class Reaction_nlin(Reaction):
    sigma:float=1.
    th:float=1.
    def ps(self,cell:Cell):#forward
        return np.tanh(cell.chemicals[self.enzymes[0]]*np.prod([cell.chemicals[i] for i in self.source])-self.th)
    def pt(self,cell:Cell):#backward
        return np.tanh(cell.chemicals[self.enzymes[1]]*np.prod([cell.chemicals[i] for i in self.target])-self.th)

@dataclass    
class Reaction_nlin1(Reaction1):
    sigma:float=1.
    th:float=1.
    def ps(self,cell:Cell):#forward
        return np.tanh(cell.chemicals[self.enzymes[0]]*cell.chemicals[self.source]-self.th)
    def pt(self,cell:Cell):#backward
        return np.tanh(cell.chemicals[self.enzymes[1]]*cell.chemicals[self.target]-self.th)
    #forward
    def ps_s(self,cells:npt.ArrayLike)->float: #NcxNm
        return np.tanh(cells[:,self.enzymes[0]]*cells[:,self.source]-self.th)
    #backward
    def pt_s(self,cells:npt.ArrayLike)->float: #NcxNm
        return np.tanh(cells[:,self.enzymes[1]]*cells[:,self.target]-self.th)


def calcEP(p:float):
    if(p>0.):
        return -p*np.log(p) 
    else:
        return 0.
    
MAXFLOAT=sys.float_info.max
warnings.simplefilter('error')

def calcepr(ps,pt):
    if(ps>pt):
        return (ps-pt)*np.log(ps/pt)
    else:
        return (pt-ps)*np.log(pt/ps)
    #assert(epr>0)

def calcReactionEPR(cell:Cell,reactions:typing.List[Reaction1])->float:
    return  np.sum(np.array([ 2*calcepr(r.ps(cell),r.pt(cell)) for r in reactions])) #反応種ごと
    
sample=lambda r=1e-8:npr.random_sample()+r
randint=npr.randint

class Cells():
    """
    Nc num. of cells
    M  num. of chemical spieces
    Nr num. of reactions
    """
    def __init__(self,Nc,M,Nr,externalchemicals,dilute="gradient",reactiontype="random",seed=42,growth=False,debug=False,sparse=True):
        npr.seed(seed)
        self.Nc=Nc
        self.M=M
        self.Nr=Nr
        self.growth=growth
        self.cells=np.array([[sample() for _ in range(M) ] for _ in range(Nc)])
        self.totsize=Nc*M
        self.externalchemicals=externalchemicals
        self.maxV=100
        self.sparse=sparse

        if(reactiontype=="random"):
            self.reactions=[ Reaction1(randint(M-1),randint(M-1),[randint(M-1),randint(M-1)]) for i in range(Nr)] #index
        elif(reactiontype=="lattice"):
            self.reactions=[ Reaction1(i,i+1,[min(i,M),max(i+1,0)]) for i in range(Nr)] #index
        elif(reactiontype=="cascade" or reactiontype=="pair"):
            self.reactions=[ Reaction1(i,i+1,[min(i+2,M),max(i-2,0)]) for i in range(0,Nr-1,2)] 
        elif(reactiontype=="pararell"): #pararell
            num=5
            Nrr=Nr-1
            Nr=Nr*num
            self.reactions=[ Reaction1(i+n*Nrr,(n*Nrr)*i+1,[max(n*Nrr+i+2,Nrr*M),min(n*Nrr+i-2,0)]) for i in range(Nrr) for n in range(num)] 
        elif(reactiontype=="forward"): #like Resnet
            self.reactions=[ Reaction1(i,i+1,[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)]             
            self.reactions+=[ Reaction1(i,min(i+5),[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)] #feed            
        elif(reactiontype=="backward"): 
            self.reactions=[ Reaction1(i,i+1,[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)]             
            self.reactions+=[ Reaction1(i,i+1,[max(i+2,M),min(i-2,0)]) for i in range(0,Nr-1,2)]  #feed                       
        else:
            print("reaction mode:random,lattice,cascade,pararell,forward,backward feedback")

        if(dilute=="gradient"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample((M-i)*0.1), "inter":sample((M-i)*0.1)} for i in range(M)]
        elif(dilute=="reverse"):# 奥(index大)のほうが大きい
            self.Ds=[{"global":sample((i+1)*0.1), "inter":sample((i+1)*0.1)} for i in range(M)]
        elif(dilute=="gradient_exp"):# 奥(index大)のほうが小さい
            self.Ds=[{"global":sample(2**((M-i)/M)), "inter":sample(2**((M-i)/M))} for i in range(M)]
        elif(dilute=="reverse_exp"):# 奥(index大)のほうが大きい
            self.Ds=[{"global":sample(2**(i/M)), "inter":sample(2**(i/M))} for i in range(M)]
        elif(dilute=="random"):
            self.Ds=[{"global":sample(), "inter":sample()} for _ in range(M)]
        elif(dilute=="constant"):
            self.Ds=[{"global":1, "inter":1.5} for _ in range(M)]
        else:
            print("diluut mode are[gradient,gradient_exp,reverse,random,constant]")

        self.inddict={"source":np.array([r.source for r in self.reactions]),
                      "target": np.array([r.target for r in self.reactions]),
                      "enzyme0":np.array([r.enzymes[0] for r in self.reactions]),
                      "enzyme1":np.array([r.enzymes[1] for r in self.reactions]),
                      }
        self.dilution_coef=np.array([d["global"] for d in self.Ds])
        self.diffusion_coef=np.array([d["inter"] for d in self.Ds])

    def calcVolume(self,ci)->float:
        return np.sum(self.cells[ci])

    def totalChemical(self)->float:#M
        return np.sum(self.cells,axis=1)

    def calcReactionEPR(self,cell):
        return np.sum(np.array([2*r.calcepr(cell) for r in self.reactions])) #scalar per a cell

    def calcTotalReactionEPR(self):
        return np.sum([self.calcReactionEPR(c) for c in self.cells])

    def calcDiffusionEPR(self):
        return self.diffusion_coef*calcepr(np.roll(self.cells,1)+np.roll(self.cells,-1)-2*self.cells)
    def calcDilutionEPR(self):
        return self.dilution_coef*calcepr(self.externalchemicals ,self.cells)
    
    def calcTotalEPR(self):
        return self.calcTotalReactionEPR()+self.calcDiffusionEPR()+self.calcDilutionEPR()
    
    def calcStaticEntropy(self):#細胞間多様性 scalar
        #totals=self.totalChemical()
        total=self.population()
        return -np.sum( np.array([calcEP(i/total) for c in self.cells for i in c ]))

    def calcStaticEntropy_chemical(self):#細胞間多様性(成分ごと) return M
        totals=self.totalChemical()
        return -np.sum( np.array([[calcEP(m/totals[i]) for i,m in enumerate(c)] for c in self.cells]),axis=1)
    
    # Total Entropy
    def calcTotalEntropy(self,dt):
        E0=self.calcStaticEntropy()
        self.run(dt)
        EPR=self.calcTotalEPR()
        E1=self.calcStaticEntropy()
        return ((E1-E0)*dt,EPR*dt)

    def calcEntropies(self,dt):
        Edif,EP=self.calcTotalEntropy(dt)
        return {"staticdif":Edif,"EPR":EP,"dif":Edif-EP}
    
    def calcEntropy_dif(self,dt):
        Ed,EP=self.calcTotalEPR(dt)
        return EP-Ed
    
    def population(self):
        return np.sum(self.cells) #[c.chemicals for c in self.cells]
    
    def dump(self,fp=sys.stdout):
        print(self.population(),file=fp)

    def run(self,dt:float=0.,debug=False):
        ncells=copy.deepcopy(self.cells) #Nc x Nm
        if(self.sparse):
            for r in self.reactions:        
                ncells[:,r.source]-=r.ps_s(self.cells)*dt
                ncells[:,r.target]+=r.pt_s(self.cells)*dt
        else: #dense
            celldict={}
            for k,ind in self.inddict: 
                 s=np.zeros_like(self.cells)
                 s[ind]=self.cells
                 celldict[k]=s
            ncells-=celldict["enzyme0"]*celldict["source"]
            ncells+=celldict["enzyme1"]*celldict["target"]

        #diffusions            
        ncells+=self.diffusion_coef*(np.roll(self.cells,1)+np.roll(self.cells,-1)-2*self.cells)*dt
        #dilutons
        ncells+=self.dilution_coef*(self.externalchemicals -self.cells)*dt

        if(self.growth):
            gamma=self.calcVolume()
            ncells=gamma*self.cells*dt
        ncells.clip(0) #数値不安定対策
        self.cells=ncells

    def _run(self,dt:float=0.,debug=False):
        ncells=copy.deepcopy(self.cells)
        N=self.Nc
        for i,c in enumerate(self.cells):
            nc=ncells[i]
            #reactions
            if(debug):
                print(f"{i}th cell")
            for p in range(len(c.chemicals)):#chemical index
                for r in self.reactions: #chemical reactions
                    if(p in r.source):
                        nc.chemicals[p]=c.chemicals[p]-r.pt(c)*dt
                    elif(p in r.target):
                        nc.chemicals[p]=c.chemicals[p]+r.ps(c)*dt
                #diffusions            
                nc.chemicals[p]+=self.Ds[p]["inter"]*(self.cells[(i-1+N)%N].chemicals[p]+self.cells[(i+1)%N].chemicals[p]-2*c.chemicals[p])*dt
                #dilutons
                nc.chemicals[p]+=self.Ds[p]["global"]*(self.externalchemicals[p] -c.chemicals[p])*dt
            if(self.growth):
                gamma=self.calcVolume(c)
                for p in range(len(c.chemicals)):
                    nc.chemicals[p]-=gamma*c.chemicals[p]*dt
        ncells.clip(0) #数値不安定対策
        self.cells=ncells

    def divide(self):
        for i,c in enumerate(self.cells):
            if(self.calcVolume(c)>self.maxV):
                ncell=copy.deepcopy(self.c)
                for p in range(len(c.chemicals)):
                    ncell.chemicals[p]+=sample()*1e-7
                self.cells.insert(i,ncell)
                self.Nc=self.Nc+1

    def initcheck(self):
        for c in self.cells:
            for i in c:    #    for i in c.chemicals:    
                assert(i>0 and i<1e6)            
        for d in self.Ds:
            assert(d["global"]>0)
            assert(d["inter"]>0)
    
    def check(self,t,msg=""):
        for i,c in enumerate(self.cells):
            for p in c:    #for p in c.chemicals:
                assert p>=0,f"negative chemical concentration t={t} {i}th,{c},{p} {msg}"
                assert p<1e6,f"chemical concentration too large t={t} {i}th,{c},{p} {msg}"            
                
    def saveparam(self,fp=sys.stdout):
        print("reactions")
        for i in  self.reactions:
            print(i.source,i.target,i.enzymes,file=fp)
        print("diffusions")
        for d in self.Ds:
            print(d["global"],d["inter"],file=fp)

    def run_all(self,T,dt=0.05,suffix="",ddir="",peri=100,debug=False,plot=True):
        if(T<peri):
            peri=T
        epss=[]
        history=[]
        totEnt=[]
        self.initcheck()
        for t in range(T):
            self.check(t)
            self.run(dt)
            if(t%peri==0):
                history.append(self.population())
                totEnt.append(self.calcEntropies(dt))

                if(self.growth):
                    self.divide()

        self.totEnt=np.array(totEnt)
        self.EP=np.array(epss)
        self.history=np.array(history).reshape(self.totsize,T//peri)
        self.save(suffix,ddir)
        if(plot):        
            self.plots(suffix,ddir)
            for p in range(self.M):
                celldist=[ c.chemicals[p] for c in self.cells]
                plt.plot(celldist)
            plt.savefig(f"{ddir}/last_cell_distribusion_{suffix}.png")            
            plt.clf()
            plt.close()

    def show(self):
        for i,c in enumerate(self.cells):
            print(f"{i}th cell:")
            print(c.chemicals)
        print("Reactions:")
        for i,r in enumerate(self.reactions):
            print(f"{i}th reaction: source:{r.source} target:{r.target} enzymes:{r.enzymes}")
        print("Diffusions:")
        for i,d in enumerate(self.Ds):
            print(f"{i}th chemical: global:{d['global']} inter:{d['inter']}")

    def plots(self,suffix,ddir):
        for k,v in {"history":self.history,"EP":self.EP,"totalEntropy":self.totEnt}.items():
            plt.plot(v)
            plt.savefig(f"{ddir}/{k}_{suffix}.png")            
            plt.clf()
            plt.close()

    def save(self,suffix,ddir):
        np.savetxt(f"{ddir}/EntropyProd_{suffix}.csv",self.EP)
        np.savetxt(f"{ddir}/totalEntropy_{suffix}.csv",self.totEnt)
        np.savetxt(f"{ddir}/history_{suffix}.csv",self.history)


def run_conds(T=1000,dt=0.01,Nc=200,M=10,r=0.6,rtype="random",dilute="gradient",growth=False,seed=0,outdir="outputs"):
    Nr=int(M*r)
    externalchemicals=[sample(seed) for _ in range(M)]
    cells=Cells(Nc,M,Nr,externalchemicals,dilute=dilute,reactiontype=rtype,seed=seed,growth=growth)
    name=f"N{Nc}_Ch{M}_r{r}_{rtype}_{dilute}_seed{seed}"
    if(growth):
        name+="_g"
    with open(f"{outdir}_params"+name+".txt","w") as fp:
        cells.saveparam(fp)

    cells.run_all(T,dt,name,ddir=outdir,peri=100)
                 
def run_allconds(T=1000,dt=0.01,Nc=200,M=10,
                 rtype=["random","lattice","cascade","pararell","forward","backward" "feedback"],
                 diftype=["gradient","gradient_exp","reverse","reverse_exp","random","constant"],
                 growth=False):
    seed=0
    for r in rtype:
        for d in diftype:
            for Mc in [M,30,50,100,500]:
                for nr in [0.1,0.5,0.8]:
                    run_conds(T,dt,Nc,Mc,nr,rtype=r,dilute=d,growth=growth,seed=seed,outdir="outputs")
                    seed+=1

def run_default(T=1000,dt=0.01,Nc=200,M=10,r="random",d="gradient",growth=False):
    run_allconds(T,dt,Nc,M,rtype=[r],diftype=[d],growth=growth)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi Cells simulator & entropy production calculation")
    parser.add_argument("--T", type=int, default=10000, help="Simulation time")
    parser.add_argument("--dt", type=float, default=0.001, help="time step")
    parser.add_argument("--N", type=int, default=200, help="num of cells")
    parser.add_argument("--M", type=int, default=100, help="num of chemical spieses")
    parser.add_argument("--r", type=float, default=0.6, help="density ratio of chemical reactions")
    parser.add_argument("--g",  action="store_true")
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--default",  action="store_true")
    args = parser.parse_args()
    if(args.all):
        run_allconds(T=args.T,dt=args.dt,Nc=args.N,M=args.M,growth=args.g)
    elif(args.default):
        run_default(T=args.T,dt=args.dt,Nc=args.N,growth=args.g)
    else:
        run_conds(T=args.T,dt=args.dt,Nc=args.N,M=args.M,r=args.r,rtype="random",dilute="gradient",growth=args.g,seed=0)    

    

                 
