
import sys
name=sys.argv[1]
out=sys.argv[2]
fr=open(name,'r')
fw=open(out,'w')

arr=fr.readlines()


for line in arr:
    
    if line.startswith("ATOM"):
       tem=line[13:16].replace(' ','')
       if not tem.startswith('H'):
            
            fw.write(line) 
    else:
        fw.write(line)
