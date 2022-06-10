cat  problem_hetatm.txt  | while read line
do


base=${line%_peptide.pdb}

mv ../$base*.pdb   .
#mv ../$line  .





done
