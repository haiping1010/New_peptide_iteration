export path1=/data/home/ZHP/rosetta/FlexPepDock_Refinement1
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