rm score_all.txt


mkdir fold_GO
for name in *.fasta

do

base=${name%.fasta}

cd $base

echo -e "$base\t\c"  >> ../score_all.txt
grep pose  model*.pdb | awk -F ' ' '{print $1, $24}' | sort -nk 2 | head -n 1  >> ../score_all.txt

line=`grep pose  model*.pdb | awk -F ' ' '{print $1, $24}' | sort -nk 2 | head -n 1`

line_n=${line:0:11 }
cp -r $line_n  ../fold_GO/$base'_'$line_n

cd ../
done


