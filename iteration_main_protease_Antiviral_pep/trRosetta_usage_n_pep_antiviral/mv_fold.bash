cat finished.txt | while read line
do

	filename=${line%/model_.pdb} 
	mv $filename'.fasta'   finished_fold

done
