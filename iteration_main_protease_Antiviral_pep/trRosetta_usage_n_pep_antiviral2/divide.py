
import sys

file=sys.argv[1]

fr=open(file,'r')
arr=fr.readlines()
index=0
for namexx in arr:
     if namexx.startswith(">"):
        
        filename=namexx.replace('>','').strip()
        fw=open(filename+'.fasta',"w")
     else:
        fw.write('>'+namexx+"\n")
        fw.write(namexx)

fw.close()

