#fr=open('summary_all.txt','r')

#arr=fr.readlines()

#name
#for name in arr:
import pandas as pd
import numpy as np
from pandas import DataFrame


aa=pd.read_csv('out_list.csv',header=0)

namelist=aa.ix[:,0].values

print (namelist[0])
basename=list(set([x.split('_complex')[0] for x in namelist]))

values=[0,]*len(basename)
y_pred=aa.ix[:,1].values
#y_vina=aa.ix[:,2].values

conf_name=[]*len(basename)

results=dict(zip(basename,values))
#vinas=dict(zip(basename,values))
conf=dict(zip(basename,conf_name))


for i in range(len(namelist)):
    base=namelist[i].split("_complex")[0]
    if y_pred[i] > results[base]:
        results[base] = y_pred[i]
        #vinas[base]=y_vina[i]
        conf[base]=namelist[i]

outcontent=[]

for i,value1 in  results.items():
        #for j,value2 in vinas.items():
        for n, value3 in conf.items():   
           if i==n:
                print i, value1, value3
                outcontent.append(i+' '+str(value1)+' '+str(value3))


outcontent=DataFrame(outcontent, columns=['all'])
new = outcontent['all'].str.split(" ", n = 2, expand = True)

outcontent2=DataFrame()

print (new)
outcontent2["name"]= new[0]
outcontent2["prediction"]= new[1].astype('float')
outcontent2["conf"]= new[2]

result=outcontent2.sort_values(by=["prediction"],ascending=False)

result.to_csv('out_list_f.csv', index = False)



