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
mkdir -p Docking/$b/$b
cd Docking/$b/$b
cp $path1/$f .
/data/program/rosetta3.5/rosetta_source/bin/docking_protocol.linuxgccrelease -database /data/program/rosetta3.5/rosetta_database $path1/@flags.lowres -s $path1/$f


cp score.sc ../score.sc > ../score0.sc
cd ..
grep -v "total_score" score0.sc > score01.sc
grep -v "SEQUENCE" score01.sc > score02.sc
sort -n -k2 score02.sc>score00.sc

#sed '1d' score00.sc>score1.sc   ##删除第一行的意思
#sed '1d' score1.sc>score2.sc

sort -n -k2 score00.sc>energy.txt   #多余

cd ../..

done


touch all_energies.list
 head -2 score.sc >> all_energies.list
cd Docking

for d in `/bin/ls`
       do
    head -1 $d/energy.txt >> ../all_energies.list
   done
cd ..
sort -k2n all_energies.list > all_energies.sort
