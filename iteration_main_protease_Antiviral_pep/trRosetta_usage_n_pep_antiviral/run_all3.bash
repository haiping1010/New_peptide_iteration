export model='/home/zhanghaiping/protein_folding/trRosetta_0'
export trRosetta='/home/zhanghaiping/protein_folding/trRosetta'

#for name in NP_*.fasta  VP_*.fasta
for name in    antiviral_{5..9}*.fasta
 
#for name in  GS*.fasta
do

base=${name%.fasta}

mkdir $base

cd $base


###############nohup  bash ../part1.bash $name  &

###############sleep 20s
####hhblits -i  ../$base'.fasta'  -o   $base'.hhr'  -oa3m  $base'.a3m' -d   $model/UniRef30_2020_06
####python $model/network/predict.py -m  $model/model2019_07   $base'.a3m'   $base'.npz'

#for id in {0..4}
#do


nohup python   $trRosetta/trRosetta.py   $base'.npz'    ../$base'.fasta'   'model_'$id'.pdb' >'model_'$id'.log' 2>&1&
sleep 3s


#done
cd ../


done



