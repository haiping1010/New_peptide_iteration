export path1=/data/home/ZHP/rosetta/FlexPepDock_Refinement
##
##this step is used to prepack
##

for w in prepack/*.pdb;do
    x=`basename $w .pdb`

cd prepack
/data/program/rosetta3.5/rosetta_source/bin/docking_protocol.linuxgccrelease -database /data/program/rosetta3.5/rosetta_database $path1/prepack/@prepack_flags -s $x".pdb"

done
cd ..
#
#this step is to run basic and lowres docking
#

for f in prepack/*_0001.pdb; do
    b=`basename $f _0001.pdb`

    echo Processing ligand $b

    mkdir -p Docking/$b
   
cd Docking/$b
cp $path1/$f .
/data/program/rosetta3.5/rosetta_source/bin/docking_protocol.linuxgccrelease -database /data/program/rosetta3.5/rosetta_database $path1/@flags.basic -s $path1/$f


cd ../..

done

