from numpy import argsort
import numpy as np
from tqdm.notebook import tqdm


class K_nn:
    def __init__(self,k=3):
        self.k=k
        
    
    def fit(self,X,y):
        self.X=X.to_numpy()
        self.y=y.to_numpy()
        
    def distance(self,x1,x2):
        return sum(pow(x1-x2,2))
        
    def predict(self,x):
        data=x.to_numpy()
        retY=[]
        with tqdm(total=len(data)) as pbar:
            for j in data:
                dist=np.sum(pow(self.X-j,2),axis=1)
                kn=self.k
                while True:
                    neighbor=self.y[argsort(dist)[:kn]]
                    occ =[neighbor.tolist().count(kk) for kk in range(10)]
                    m= max(occ)
                    if (occ.count(m)==1):
                        retY.append(occ.index(m))
                        break
                    kn+=1
                pbar.update(1)
        return retY
    
    def set_params(self,**arg):
        self.k=arg["k"]
        print(self.k)
        

            
            
        
        
        
        
        
        
            
       