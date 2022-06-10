export path1=/data/home/ZHP/rosetta/FlexPepDock_Refinement1/PD1
cd /data/home/ZHP/rosetta/FlexPepDock_Refinement1/PD1
cat  */all_energies*sort >aa.txt



grep -v "total_score" aa.txt > aaa.txt
grep -v "SEQUENCE" aaa.txt > bbb.txt
sort -n -k2 bbb.txt > score00.sort

awk -F ' ' '{print $2 "  " $27 }' score00.sort > score000.sort 
