
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

sample=lambda r=1e-8:npr.random_sample()+r
randint=npr.randint

MAXFLOAT=sys.float_info.max
warnings.simplefilter('error')

def calcepr(ps:float,pt:float):
    d=np.abs(ps-pt)
    pp=(d>0).astype(int)
    return pp*(-d*np.log(d+1e-10))
    
def calcEP(p:float):
    pp=(p>0).astype(int)
    return pp*(-p*np.log(p+1e-10))

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
class Reaction_nlin(Reaction):
    sigma:float=1.
    th:float=1.
    def ps(self,cell:Cell):#forward
        return np.tanh(cell.chemicals[self.enzymes[0]]*np.prod([cell.chemicals[i] for i in self.source])-self.th)
    def pt(self,cell:Cell):#backward
        return np.tanh(cell.chemicals[self.enzymes[1]]*np.prod([cell.chemicals[i] for i in self.target])-self.th)

@dataclass    
class Reaction1():
    source:int #index
    target:int #index
    enzymes:typing.List[int] #index
    def __init__(self,s,t,enz):
        self.source=s
        self.target=t
        self.enzymes=enz
    #forward
    def ps_c(self,cell:npt.ArrayLike)->float: 
        return cell[self.enzymes[0]]*cell[self.source]
    #backward
    def pt_c(self,cell:npt.ArrayLike)->float: 
        return cell[self.enzymes[1]]*cell[self.target]

    def ps_cells(self,cells:npt.ArrayLike)->float: #NcxNm
        return cells[:,self.enzymes[0]]*cells[:,self.source]

    def pt_cells(self,cells:npt.ArrayLike)->float: #NcxNm
        try:
            return cells[:,self.enzymes[1]]*cells[:,self.target]
        except:
            print(cells)
            raise
    def calcepr(self,cell):
        return calcepr(self.ps_c(cell),self.pt_c(cell))

@dataclass    
class Reaction_nlin1(Reaction1):
    sigma:float=1.
    th:float=1.
    #forward
    def ps_c(self,cell:npt.ArrayLike)->float: 
        return np.tanh(self.sigma*cell[self.enzymes[0]]*cell[self.source]-self.th)
    #backward
    def pt_c(self,cell:npt.ArrayLike)->float: 
        return np.tanh(self.sigma*cell[self.enzymes[1]]*cell[self.target]-self.th)

    #forward
    def ps_cells(self,cells:npt.ArrayLike)->float: #NcxNm
        return np.tanh(self.sigma*cells[:,self.enzymes[0]]*cells[:,self.source]-self.th)
    #backward
    def pt_cells(self,cells:npt.ArrayLike)->float: #NcxNm
        return np.tanh(self.sigma*cells[:,self.enzymes[1]]*cells[:,self.target]-self.th)


class Cells():
    """
    Nc num. of cells
    M  num. of chemical spieces
    Nr num. of reactions
    """
    def __init__(self,Nc,M,Nr,externalchemicals,dilute="gradient",reactiontype="random",
                 seed=42,growth=False,debug=False,sparse=True):
        npr.seed(seed)
        self.orgCellNum=Nc
        self.Nc=Nc
        self.M=M
        self.Nr=Nr
        self.growth=growth
        self.cells=np.array([[sample() for _ in range(M) ] for _ in range(Nc)])
        self.totsize=Nc*M
        self.externalchemicals=externalchemicals
        self.maxV=100
        self.sparse=sparse
        self.name=""f"N{Nc}_Ch{M}_Nr{Nr}_{reactiontype}_{dilute}_seed{seed}"

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

        with open(f"{self.name}_init.txt","w") as fp:
            self.printinit(fp)            

        self.growthconst=0.1
        self.menbramechemical=self.M//2
        self.linage=[i for i in range(Nc)]
        
    def population(self):
        return np.sum(self.cells) #[c.chemicals for c in self.cells]
    def Volume(self)->float: #Nc
        return np.sum(self.cells,axis=1)
    def cellVolume(self,cell)->float: #Nc
        return np.sum(cell)
    def totalChemical(self)->float:#M
        return np.sum(self.cells,axis=0)

    def ReactionEPR(self,cell):
        return np.sum(np.array([2*r.calcepr(cell) for r in self.reactions])) #scalar per a cell
    
    def TotalReactionEPR(self):
        return np.sum([self.ReactionEPR(c) for c in self.cells])

    def DiffusionEPR(self):
        return np.sum(self.diffusion_coef*calcepr(np.roll(self.cells,1)+np.roll(self.cells,-1),2*self.cells))
    def DilutionEPR(self):
        return np.sum(self.dilution_coef*calcepr(self.externalchemicals ,self.cells))
    
    def TotalEPR(self):
        return self.TotalReactionEPR()+self.DiffusionEPR()+self.DilutionEPR()
    
    def StaticEntropy(self):#細胞間多様性 scalar
        assert(np.all(self.cells>=0))
        return np.sum( np.array([calcEP(i/self.totals) for c in self.cells for i in c ]))

    def StaticEntropy_Cell(self):#細胞間多様性 scalar
        assert(np.all(self.cells>=0))
        return np.sum( np.array([calcEP(i/self.totals) for c in self.cells for i in c ]))

    def StaticEntropy_chemical(self):#細胞間多様性(成分ごと) return M
        totals=self.totalChemical()
        return np.sum( np.array([[calcEP(m/totals[i]) for i,m in enumerate(c)] for c in self.cells]),axis=1)
    
    # Total Entropy
    def TotalEntropy(self,dt):
        E0=self.StaticEntropy()
        self.run(dt)
        EPR=self.TotalEPR()
        self.cells=self.cells.clip(0)
        E1=self.StaticEntropy()
        return ((E1-E0)*dt,EPR*dt)

    def Entropies(self,dt):
        Edif,EP=self.TotalEntropy(dt)
        return {"staticdif":Edif,"EP":EP,"dif":Edif-EP}
    
    def run(self,dt:float=0.,debug=False):
        ncells=copy.deepcopy(self.cells) #Nc x Nm
        if(self.sparse):
            for r in self.reactions:        
                ncells[:,r.source]-=r.ps_cells(self.cells)*dt #
                ncells[:,r.target]+=r.pt_cells(self.cells)*dt
        else: #dense
            celldict={}
            for k,ind in self.inddict: 
                 s=np.zeros_like(self.cells)
                 s[ind]=self.cells
                 celldict[k]=s
            ncells-=celldict["enzyme0"]*celldict["source"]
            ncells+=celldict["enzyme1"]*celldict["target"]

        ncells+=self.diffusion_coef*(np.roll(self.cells,1)+np.roll(self.cells,-1)-2*self.cells)*dt
        ncells+=self.dilution_coef*(self.externalchemicals -self.cells)*dt

        if(not self.growth):#等倍縮小 (V-self.V)/dt:=growthrate
            growthrate=self.growthconst*self.cells[:,self.menbramechemical]
            ncells-=(growthrate.reshape(growthrate.shape[0],1)*self.cells)*dt
            
        self.cells=ncells.clip(0) #数値不安定対策

    def divide(self,mabiki=10):
        for i,c in enumerate(self.cells):
            if(self.cellVolume(c)>self.maxV):
                ncell=copy.deepcopy(c)
                for p in range(len(c)):
                    ncell[p]+=sample()*1e-7
                np.insert(self.cells,i,ncell)
                np.insert(self.linage,i,i)
                self.Nc=self.Nc+1
        if(self.Nc>self.orgCellNum*mabiki):#mabiki
            self.cells=self.cells[::mabiki,:]
            self.linage=self.linage[::mabiki]
            self.Nc=self.cells.shape[0]

    def initcheck(self):
        for c in self.cells:
            for i in c:    #    for i in c.chemicals:    
                assert(i>0 and i<1e6)            
        for d in self.Ds:
            assert(d["global"]>0)
            assert(d["inter"]>0)

    def saveparam(self,fp=sys.stdout):
        print("reactions",file=fp)
        for i in  self.reactions:
            print(i.source,i.target,i.enzymes,file=fp)
        print("diffusions",file=fp)
        for d in self.Ds:
            print(d["global"],d["inter"],file=fp)

    def run_all(self,T,dt=0.05,suffix="",ddir="",peri=100,debug=False,plot=True):
        if(T<peri):
            peri=T
        history=[]
        totEnt=[]
        volumehistory=[]
        self.V=self.Volume()
        self.initcheck()
        for t in range(T):
            self.run(dt)
            if(np.any(self.cells<0)):
                print(f"negative chemical concentration t={t}")
                break
            elif(np.any(self.cells>1e8)):
                print(f"chemical concentration too large t={t}")
                break
            if(t%peri==0):
                total=self.population()
                self.totals=self.totalChemical()
                assert(np.all(self.totals>=0))
                ent=self.Entropies(dt)
                history.append(total)
                volumehistory.append(self.V)
                totEnt.append(ent)
                if(debug):
                    print(f"{t}: total {total}, chemicals{self.totals},ent {ent}")
                if(self.growth):
                    self.divide()
        self.totEnt=totEnt
        self.history=np.array(history) 
        self.volumehistory=np.array(volumehistory) 
        self.save(suffix,ddir)
        with open(f"{self.name}_last.txt","w") as fp:
            print(f"{self.name}, total population: {total}, Volumes: {self.Volume()},total chemicals: {self.totals},ent: {ent}",file=fp)

        if(plot):        
            self.plots(suffix,ddir)
            for p in range(self.M):
                celldist=[ c[p] for c in self.cells]
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
        for k,v in {"total_population":self.history,"EP":self.EP,"staticEntropy":self.staticEntropy}.items():
            plt.plot(v)
            plt.savefig(f"{ddir}/{k}_{suffix}.png")            
            plt.clf()
            plt.close()

    def save(self,suffix,ddir):
        tmp={"staticdif":[],"EP":[],"dif":[]}
        for l in self.totEnt:
            for k in tmp:
                tmp[k].append(l[k])
        for k in tmp:                
            tmp[k]=np.array(tmp[k])
            try:
                np.savetxt(f"{ddir}/totalEntropy_{k}_{suffix}.csv",tmp[k])
            except:
                print(k,tmp[k].shape)
        self.EP=tmp["EP"]
        self.staticEntropy=tmp["staticdif"]
        np.savetxt(f"{ddir}/total_population_{suffix}.csv",self.history)
        if(self.growth):
            np.savetxt(f"{ddir}/linage_{suffix}.csv",np.array(self.linage))
        
    def printinit(self,fp=sys.stdout):
        print("param: ",self.name,file=fp)
        print("Cells shape: ",self.cells.shape,file=fp)
        print("total population: ",self.population(),file=fp)
        print("Cell  Volume: ",self.Volume(),file=fp)
        print("Total Chemicals: ",self.totalChemical(),file=fp)
#class

def run_conds(T=1000,dt=0.01,Nc=200,M=10,r=0.6,rtype="random",dilute="gradient",growth=False,seed=0,outdir="outputs",debug=False,peri=100):
    Nr=int(M*r)
    externalchemicals=[sample(seed) for _ in range(M)]
    cells=Cells(Nc,M,Nr,externalchemicals,dilute=dilute,reactiontype=rtype,seed=seed,growth=growth)
    name=f"N{Nc}_Ch{M}_r{r}_{rtype}_{dilute}_seed{seed}"
    if(growth):
        name+="_g"
    with open(f"{outdir}/params"+name+".txt","w") as fp:
        cells.saveparam(fp)

    cells.run_all(T,dt,name,ddir=outdir,peri=peri,debug=debug)
                 
def run_allconds(T=1000,dt=0.01,M=10,
                 rtype=["random","lattice","cascade","pararell","forward","backward" "feedback"],
                 diftype=["gradient","gradient_exp","reverse","reverse_exp","random","constant"],
                 growth=False,
                 peri=100):
    seed=0
    for r in rtype:
        for d in diftype:
            for Nc in [100,200,400,500]:
                for Mc in [M,30,50,100,500]:
                    for nr in [0.1,0.5,0.8]:
                        run_conds(T,dt,Nc,Mc,nr,rtype=r,dilute=d,growth=growth,seed=seed,outdir="outputs",peri=peri)
                        seed+=1

def run_default(T=1000,dt=0.01,M=10,r=["random"],d=["gradient"],growth=False):
    run_allconds(T,dt,M,rtype=r,diftype=d,growth=growth)

def run_N(T=1000,dt=0.01,Nc=200,M=10,growth=False):
    rtype=["random","lattice","cascade","pararell","forward","backward" "feedback"],
    diftype=["gradient","gradient_exp","reverse","reverse_exp","random","constant"],
    seed=0
    for growth in [True,False]:
        for r in rtype:
            for d in diftype:
                for Mc in [M,30,50,100,500]:
                    for nr in [0.1,0.5,0.8]:
                        run_conds(T,dt,Nc,Mc,nr,rtype=r,dilute=d,growth=growth,seed=seed,outdir="outputs")
                        seed+=1

def run_small(T,dt,rtype=["random","cascade"],
              diftype=["gradient","reverse","random"],growth=False):
    seed=0
    for r in rtype:
        for d in diftype:
            for Nc in [10,20,50]:
                for Mc in [10,20,50]:
                    for nr in [0.1,0.3,0.5]:
                        run_conds(T,dt,Nc,Mc,nr,rtype=r,dilute=d,growth=growth,seed=seed,outdir="outputs")
                        seed+=1

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi Cells simulator & entropy production calculation")
    parser.add_argument("--T", type=int, default=10000, help="Simulation time")
    parser.add_argument("--dt", type=float, default=0.001, help="time step")
    parser.add_argument("--N", type=int, default=200, help="num of cells")
    parser.add_argument("--M", type=int, default=100, help="num of chemical spieses")
    parser.add_argument("--r", type=float, default=0.6, help="density ratio of chemical reactions")
    parser.add_argument("--peri", type=int, default=100, help="period to calualate entoropy")
    parser.add_argument("--g",  action="store_true",help="growth mode/divide mode")
    
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--default",  action="store_true")
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--small",  action="store_true")


    args = parser.parse_args()
    if(args.all):
        run_allconds(T=args.T,dt=args.dt,M=args.M,growth=args.g)
    elif(args.default):
        run_default(T=args.T,dt=args.dt,growth=True)
#        run_default(T=args.T,dt=args.dt,growth=False)
    elif(args.small):
        run_small(T=args.T,dt=args.dt,growth=False)    
        run_small(T=args.T,dt=args.dt,growth=True)    
    else:
        run_conds(T=args.T,dt=args.dt,Nc=args.N,M=args.M,r=args.r,rtype="random",dilute="gradient",growth=args.g,seed=0,debug=True)    

    

                 
