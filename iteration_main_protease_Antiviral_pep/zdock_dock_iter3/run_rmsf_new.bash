
for m in ????.pdb
do
i=${m%.pdb}



echo 1 1 | g_rmsf -s $i".pdb" -f $i".pdb"  -fit  -ox $i'xaver.pdb'
echo 1 1 | g_rms -s $i'xaver.pdb' -f $i".pdb"  -fit rot+trans -o $i'avrms.xvg' 



grep -v "^#\|^@" $i'avrms.xvg' > file.txt
min=1000
while read LINE
do
    fir=`echo $LINE|awk '{print $1}'`
    sec=`echo $LINE|awk '{print $2}'`
st=`echo "$sec < $min" | bc`
    if [ $st -eq 1 ];then
        min=$sec
        minindex=$fir
    fi
done < <(cat file.txt)

minindexint=${minindex%.*}
echo 1 | trjconv -s $i".pdb" -f $i".pdb"  -o $i"_"$minindexint'av_rep.pdb' -b $minindex -e $minindex 


cp -r $i"_"$minindexint'av_rep.pdb' $i"_n.pdb"



#echo 36 37 | /home/zhp/program/gromacs-4.6.5/bin/bin/g_dist -s md2.tpr -f md2.xtc -o $i'domain.xvg' -n index.ndx

#cp $i'xaver.pdb' $i"_"$minindexint'av_rep.pdb' $i'avrms.xvg'  ../AVRAGE


done
