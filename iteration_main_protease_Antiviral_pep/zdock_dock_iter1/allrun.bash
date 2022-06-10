
for f in prepack/*_0001.pdb; do
    b=`basename $f _0001.pdb`

    echo Processing ligand $b
mkdir -p Dockingx/$b
  done