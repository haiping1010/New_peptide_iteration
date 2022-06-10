export model='/home/zhanghaiping/protein_folding/trRosetta_0'
export trRosetta='/home/zhanghaiping/protein_folding/trRosetta'




name=$1

base=${name%.fasta}

#echo $base
hhblits -i  ../$base'.fasta'  -o   $base'.hhr'  -oa3m  $base'.a3m' -d   $model/UniRef30_2020_06

python $model/network/predict.py -m  $model/model2019_07   $base'.a3m'   $base'.npz'

