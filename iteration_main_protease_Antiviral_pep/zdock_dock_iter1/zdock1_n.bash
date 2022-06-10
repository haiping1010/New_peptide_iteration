export path=/home/zhanghaiping/program/zdock3.0.2_linux_x64
: '
cp -r xxxx_w.pdb  receptor.pdb
$path/mark_sur_static receptor.pdb receptor_m.pdb


for w in ligands/???_seq.pdb
  do
    #x=`basename $w .pdb`
    x=${w:8 }
    x=${x%.pdb}
    echo $x"_m.pdb"
    sed -i 's/ HIE / HIS /'   $w
    sed -i 's/ HIP / HIS /'   $w
    python  rm_h.py $w  ligands/$x"_w.pdb" 
    $path/mark_sur_static  ligands/$x"_w.pdb"  ligands/$x"_m.pdb"

done
'
#mkdir Docking
#mkdir result
for f in ligands/model_*_m.pdb; do
    b=`basename $f _m.pdb`
    echo Processing ligand $b
    mkdir -p Docking/$b
   
    #nohup $path/zdock -R  receptor_m.pdb -L  ligands/$b'_m.pdb' -o  Docking/$b/zdock.out  > out.log 2>&1&
#sleep 5s


rm -rf complex.1.pdb
perl $path/create.pl  Docking/$b/zdock.out 1
mv complex.1.pdb   Docking/$b/$b"_complex.pdb"
cp -r Docking/$b/$b"_complex.pdb"   result/$b"_complex.pdb"

done

