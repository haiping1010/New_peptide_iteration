

for i in *.pdb
    do
f=${i%.pdb}
mkdir $f
cd $f
/data/home/ZHP/rosetta3.4/rosetta_source/bin/FlexPepDocking.linuxgccrelease -database /data/home/ZHP/rosetta3.4/rosetta_database @/data/home/ZHP/rosetta/flexpepdock/flags > flexpepdock.log

sort -n -k2 score.sc>energy.txt

cd ..
done



