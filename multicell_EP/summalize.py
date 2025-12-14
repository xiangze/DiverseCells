import json
import numpy as np
import glob
import statistics as st
import pandas as pd

def getparam(f):
    params=f.split("_")[:-1]
    return [p.replace("Nr","").replace("N","").replace("Ch","").replace("seed","") for p in params]

def parse():
    nds=[]
    for i,f in enumerate(glob.glob("outputs/summary*txt")):
        with open(f) as fp:
            d=json.load(fp)
        nd={}
        for ii in ["init","last"]:
            for k,v in d["init"].items():
                if(not "Total " in k and not "Volume" in k and k!="ent"):
                    nd[f"{k}_{ii}"]=v
            for k,v in d[ii]["ent"].items():
                    nd[f"{k}_{ii}"]=v
            for l in ["Cell  Volume","Total Chemicals"]:
                nd[f"{l}_max_{ii}"]=max(d[ii][l])
                nd[f"{l}_min_{ii}"]=min(d[ii][l])
                nd[f"{l}_ave_{ii}"]=st.mean(d[ii][l])
                nd[f"{l}_var_{ii}"]=st.pvariance(d[ii][l])
        nds.append(nd)
    df=pd.DataFrame(nds)        
    df.to_csv("summary_div.csv")

if __name__=="__main__":
    parse()        