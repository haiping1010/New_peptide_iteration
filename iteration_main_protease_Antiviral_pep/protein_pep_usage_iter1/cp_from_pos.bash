mkdir result_f
cat   list_w.txt  | while read line
 do

IFS=',' read -r -a array <<< $line


base=${array[0]}
base_n=${base:6 }

echo $base_n
cp -r  'antivirus/'$base_n'.fasta'  result_f


done 

cd result_f
cat antiviral_*.fasta | grep -v '^$\|>' > all.fasta

cd ../




