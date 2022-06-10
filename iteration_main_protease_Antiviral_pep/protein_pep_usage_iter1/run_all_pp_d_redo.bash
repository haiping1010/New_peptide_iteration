
cd docking_complex/

for name in *_L.pdb
do
#echo $LINE
base=${name%_L.pdb}








###cd original_PDB
nohup python   ../python_2_pp_sep_d.py  $base &
sleep  0.8s


#echo $LINE

done

