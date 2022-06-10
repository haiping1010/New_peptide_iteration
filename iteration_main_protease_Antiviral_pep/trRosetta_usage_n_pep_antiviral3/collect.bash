mkdir collect
for name in *.fasta
do

	base=${name%.fasta}
cp -r $base/model_.pdb   collect/'model_'$base'.pdb'

done
