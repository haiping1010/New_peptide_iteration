export model='/home/zhanghaiping/protein_folding/trRosetta_0'
export trRosetta='/home/zhanghaiping/protein_folding/trRosetta'

#for name in NP_*.fasta  VP_*.fasta

#for name in  GS*.fasta




python $model/network/predict_many.py -m  $model/model2019_07   many2   many2

#for id in {0..4}
#do


#nohup python   $trRosetta/trRosetta.py   $base'.npz'    ../$base'.fasta'   'model_'$id'.pdb' >'model_'$id'.log' 2>&1&
#sleep 10s


#done


done



