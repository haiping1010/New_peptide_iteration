export model='/home/zhanghaiping/protein_folding/trRosetta_0'
export trRosetta='/home/zhanghaiping/protein_folding/trRosetta'

#for name in NP_*.fasta  VP_*.fasta
for name in    *.fasta

#for name in  GS*.fasta
do

base=${name%.fasta}

mkdir $base

cd $base



#  python  /home/zhanghaiping/protein_folding/trRosetta_0/network/predict.py -m  /home/zhanghaiping/protein_folding/trRosetta_0/model2019_07  aa.a3m  aa.npz

hhblits -i  ../$base'.fasta'  -o   $base'.hhr'  -oa3m  $base'.a3m' -d   $model/UniRef30_2020_06

python $model/network/predict.py -m  $model/model2019_07   $base'.a3m'   $base'.npz'

#for id in {0..4}
#do


nohup python   $trRosetta/trRosetta.py   $base'.npz'    ../$base'.fasta'   'model_'$id'.pdb' >'model_'$id'.log' 2>&1&
sleep 10s


#done
cd ../


done



