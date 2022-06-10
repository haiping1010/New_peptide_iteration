cd Docking
for d in `/bin/ls`
       do
sort -k2n $d/score.sc > $d/energy
    head -1 $d/energy >> ../all_energies.list
   done
cd ..
sort -k2n all_energies.list > all_energies.sort