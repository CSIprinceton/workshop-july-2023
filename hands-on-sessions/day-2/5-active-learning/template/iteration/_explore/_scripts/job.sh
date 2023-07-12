date
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
for i in `seq 0 100`
do
	echo "structure $i"
     	mpirun -n 1 $PW -input pw-si-$i.in > pw-si-$i.out
done
date
