
import sys
filename=sys.argv[1]

fr=open(filename,'r')


arr=fr.readlines()
old_id=0
old_arr=[]
tem_arr=[]
for name  in arr:
    if name.startswith('ATOM'):
       id=name[22:26].replace(' ','')
       #tem_arr.append(name)
       if int(old_id) !=0 and  int(id) != int(old_id) and int(id) != int(old_id)+1:
           print old_id
           print name
           old_arr.extend(tem_arr)
           tem_arr=[]
       tem_arr.append(name)

       old_id=name[22:26].replace(' ','')


fw_r=open(filename.replace('complex1.pdb','R.pdb'),'w')
fw_l=open(filename.replace('complex1.pdb','L.pdb'),'w')


for name in old_arr:
    fw_r.write(name)    

fw_r.close()

for name in tem_arr:
    fw_l.write(name)

fw_l.close()

       #print old_id
