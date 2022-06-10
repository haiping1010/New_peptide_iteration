import sys

file1=sys.argv[1]
file2=sys.argv[2]

mapall={}
fr=open(file1,'r')
arr_fr=fr.readlines()
for name in arr_fr:
   if name.startswith('ATOM'):
      name=list(name)  
      index=name[17:26]
      value=name[55:57]
      index=''.join(index)
      value=''.join(value)
      mapall[index]=value
      #print mapall[index]
      #print value

#for k, v in mapall.items():
#    print(k, v)

fr2=open(file2,'r')
fw=open(file2.replace('_m.pdb','_mm.pdb'),'w')
arr_fr2=fr2.readlines()
for name in arr_fr2:
   if name.startswith('ATOM'):
      name=list(name)
      index=name[17:26]
      value=name[55:57]
      index=''.join(index)
      if  mapall[index]=='19':
          name[55:57]='19'
          fw.write(''.join(name))
      else:
          fw.write(''.join(name))
   else:
      fw.write(''.join(name))







