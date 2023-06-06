for i in 1 2 3 4
do
	cd $i
	dp freeze
	mv frozen_model.pb frozen_model_$i.pb
	cd ..
done
