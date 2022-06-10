import sys
import numpy as np

      
visited = {}
HLposition={}
Eposition={}
ResinameHL={}
ResinameE={}
antiface=[]
residuePair=[]
if len(sys.argv) <2 :
   print("python python2.py xxx.pdb")
file=sys.argv[1]
file2=sys.argv[2]

filebase=file.replace("_n.pdb","")


for line in open(file):
    list = line.split()
    id = list[0]
    if id == 'ATOM':
            residue = list[3]
            type_of_chain = line[21:22]
			
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem2.replace("C", "")

            #tem2=filter(str.isdigit, list[5])
            #atom_count = tem2+list[4]
            atom_count = tem2+line[21:22]
            list[6]=line[30:38]
            list[7]=line[38:46]
            list[8]=line[46:54]
            position = list[6:9]
            HLposition[atom_count]=position
            ResinameHL[atom_count]=residue


for line in open(file2):
    list = line.split()
    id = list[0]
    if id[0:6] == 'HETATM':
            ###becareful
            type = line[13:16].replace(' ','')
            residue = line[17:20]
            print (type,residue)
            type_of_chain = line[21:22]
            tem1=line[22:26].replace("A", "")
            tem2=tem1.replace("B", "")
            tem2=tem1.replace("C", "")

            #tem2=filter(str.isdigit, list[5])
            atom_count = tem2+line[21:22]
            list[6]=line[30:38]
            list[7]=line[38:46]
            list[8]=line[46:54]
            position = list[6:9]
            Eposition[atom_count]=position



for key1, value1 in HLposition.items():
   for key2, value2 in Eposition.items():
            #print (ResinameE[key], 'corresponds to', value)
            ##distance=pow(value1[0]-value2[0])
            #print key2
            a = np.array(value1)
            a1 = a.astype(np.float)
            b = np.array(value2)
            b1 = b.astype(np.float)
            xx=np.subtract(a1,b1) 
            tem=np.square(xx)
            tem1=np.sum(tem)
            out=np.sqrt(tem1)
            #print (out)
            if out<12 :
                residuePair.append(key1)
                #residuePair.append(key2)
                antiface.append(a1)
                #print (a1)
                #print (antiface)              


for name in residuePair:
    print name
#print (antiface)              
antibody = open("receptor_mm.pdb", "w")
for line in open(file):
    list = line.split()
    if list[0] == 'ATOM'   :
        temstr=line[22:26]
        if   line[22:26] + line[21:22] in  residuePair:
            antibody.write(line)  
            #print line
        else: 
            line=line[:55] + '19'+ line[57:] 
            antibody.write(line) 
    elif list[0] == 'TER'  :	
                  antibody.write(line)  
