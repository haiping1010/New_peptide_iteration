export path=/home/zhanghaiping/program/zdock3.0.2_linux_x64
: '
cp -r  6y2f_w.pdb   receptor.pdb


$path/mark_sur_static receptor.pdb receptor_m.pdb

python python_block_n.py  receptor_m.pdb   6y2f_ligand_n.pdb

##rm -f ligands/model_*_m.pdb   ligands/model_*_w.pdb
for w in ligands/model_*.pdb
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
mkdir Docking

mkdir result
#for f in    ligands/model_antiviral3_38{5..9}*_m.pdb; 
cat   need_rerun.txt  | while   read   b

do
    echo Processing ligand $b
    mkdir -p Docking/$b
   
    nohup $path/zdock -R  receptor_mm.pdb -L  ligands/$b'_m.pdb' -o  Docking/$b/zdock.out -N 50 > out.log 2>&1&
    sleep  7s
: '
rm -rf complex.1.pdb
perl $path/create.pl  Docking/$b/zdock.out 1
mv complex.1.pdb   Docking/$b/$b"_complex.pdb"
cp -r Docking/$b/$b"_complex.pdb"   result/$b"_complex.pdb"
'
done

