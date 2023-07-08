conda deactivate
export PW=/home/deepmd23admin/Softwares/QuantumEspresso/q-e-qe-7.0/bin/pw.x
for i in `seq 0 99`
do
        mpirun -np 1 $PW -input pw-si-$i.in > pw-si-$i.out
done