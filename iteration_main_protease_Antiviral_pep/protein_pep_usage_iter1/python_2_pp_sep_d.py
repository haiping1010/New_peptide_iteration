import sys
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
def one_hot(i):
    m = np.zeros(20, 'uint8')
    m[i] = 1
    return m

def MapResToOnehot(residue1,residue2):
   code={}
   seq=("ALA","ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL")
   index=0
   for name in seq:
              #print (name)
              code[name]=one_hot(index)
              index=index+1
   return (np.concatenate((code[residue1],code[residue2])))
#print (MapResToOnehot("ARG","VAL"))

      
visited = {}
HLposition={}
Eposition={}
ResinameHL={}
ResinameE={}
antiface=[]
residuePair=[]
if len(sys.argv) <1 :
   print("python python2.py xxx.pdb")
file=sys.argv[1]
filebase=file
import glob
anti=glob.glob(file+"*_L.pdb")
print(file)
for line in open(anti[0]):
    tem_B=' '
    if len(line)>16:
       tem_B=line[16]
       line=line[:16]+' '+line[17:]
    #print(line)
    list = line.split()
    id = list[0]
    if id == 'ATOM' and tem_B !='B':
        type = list[2]
        if type == 'CA'  and list[3]!= 'UNK'  :
            residue = list[3]
            type_of_chain = line[21:22]
			
            ##tem1=list[5].replace("A", "")
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem2.replace("C", "")

            #tem2=filter(str.isdigit, list[5])
            atom_count = tem2+line[21:22]
            list[6]=line[30:38]
            list[7]=line[38:46]
            list[8]=line[46:54]
            position = list[6:9]
            HLposition[atom_count]=position
            ResinameHL[atom_count]=residue

antig=glob.glob(file+"*_R.pdb")

#print antig[0]
for line in open(antig[0]):
    tem_B=' '
    if len(line)>16:
       tem_B=line[16]
       line=line[:16]+' '+line[17:]
    list = line.split()
    id = list[0]
    if id == 'ATOM' and tem_B !='B':
        type = list[2]
        if   type == 'CA' and list[3]!= 'UNK':
            residue = list[3]
            type_of_chain = line[21:22]
            tem1=list[5].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem1.replace("C", "")

            #tem2=filter(str.isdigit, list[5])
            atom_count = tem2+line[21:22]
            list[6]=line[30:38]
            list[7]=line[38:46]
            list[8]=line[46:54]
            position = list[6:9]
            Eposition[atom_count]=position
            ResinameE[atom_count]=residue
            #print position

for key1, value1 in HLposition.items():
   #print (ResinameHL[key1], 'corresponds to', value1)
   for key2, value2 in Eposition.items():
            #print key2,value2
            #print (ResinameE[key2], 'corresponds to', value2)
            ##distance=pow(value1[0]-value2[0])
            a = np.array(value1)
            a1 = a.astype(np.float)
            b = np.array(value2)
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1) 
            tem=np.square(xx)
            tem1=np.sum(tem)
            out=np.sqrt(tem1)
            #print a
            if out<10 :
                #print (ResinameHL[key1],ResinameHL[key2])
                #print (a,b,out)
                fo = open("ceshi.txt", "wb")
                fo.write(a)
                fo.write(b)
                #print (a1)              
                residuePair.append([ResinameHL[key1],ResinameE[key2]])
                antiface.append(a1)
                #print (antiface)              
                

#print (antiface)              
kmeans = KMeans(n_clusters=5, random_state=0).fit_predict(antiface)
#print (kmeans)
indexx=0
kmeans1=kmeans.tolist()
foo = open(filebase + "_learn.txt", "w")
index_n=0
for i in range(5):
   #print(kmeans)
   g=np.argwhere(kmeans==i)
   ##g=kmeans.where(q==i) 
   ##print (g)
   g1=g.reshape(-1)

   for groupid in g1:
      #print (groupid)
      #print (residuePair[groupid])
      #foo = open("3HFM_learn.txt", "a")
      #foo.write(residuePair[groupid][0])
      #foo.write(residuePair[groupid][1])
      index_n=index_n+1
      if index_n<=300:
           writout=MapResToOnehot(residuePair[groupid][0],residuePair[groupid][1])
           # foo = open("3HFM_learn.txt", "a")
           arr_tem=np.array_str(writout)
           for code in  writout:
              #print (str(code))
              foo.write(str(code))
              # print(code)
              #foo.write(code)
           foo.write('\n')
while index_n<300:
         index_n=index_n+1
         foo.write("0000000000000000000000000000000000000000\n")

#print (residuePair[1][0])
#for i in residuePair:
#    print (i)

  
##     print (residue,type_of_chain,atom_count,position)

