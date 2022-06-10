for name in *_receptor.pdb
do
#echo $LINE
base=${name:0:6}








###cd original_PDB
nohup python   python_2_pp_sep.py  $base &
sleep  0.5s


#echo $LINE

done

