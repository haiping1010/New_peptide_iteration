export model='/home/zhanghaiping/protein_folding/trRosetta_0'
export trRosetta='/home/zhanghaiping/protein_folding/trRosetta'

#for name in NP_*.fasta  VP_*.fasta
for name in     antiviral4_*.fasta

#for name in  GS*.fasta
do

base=${name%.fasta}

mkdir $base

cd $base



nohup hhblits -i  ../$base'.fasta'  -o   $base'.hhr'  -oa3m  $base'.a3m' -d   $model/UniRef30_2020_06 &
sleep 3s


cd  ../



done



