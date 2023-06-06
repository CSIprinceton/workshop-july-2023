counter=0
for seed in 25875 31473 26182 12396
do
	counter=$(($counter+1))
	cp -r BASE $counter
	cd $counter
	sed -i "s/SEED/$seed/g" input.json
	sed -i "s/SEED/$seed/g" input-compress.json
	sed -i "s/REPLACE/$counter/g" job.sh
	sed -i "s/REPLACE/$counter/g" job-compress.sh
	sbatch < job.sh
	cd ..
done
