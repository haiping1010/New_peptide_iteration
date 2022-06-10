
import sys

input_f=sys.argv[1]

fr=open(input_f,'r')
arr=fr.readlines()

fw=open(input_f.replace('_gen_3.fa','.fasta'),'w')



index=0

for name in arr:
    print name
    fw.write(  '>'+input_f.replace('_gen_3.fa','_')    +str(index))   
    fw.write("\n")
    fw.write(name)
    index=index+1

fw.close()

