export path1=/data/home/ZHP/rosetta/FlexPepDock_Refinement1/FlexPepDock_Refinement


for f in prepack/*_0001.pdb; do
    b=`basename $f _0001.pdb`

    echo Processing ligand $b


   


cd /data/home/ZHP/rosetta/FlexPepDock_Refinement1/FlexPepDock_Refinement/Docking/$b/$b


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
