echo "Testing Viola-Jones CUDA with several scales of the Solvay conference image."
iterations=10
for scale in solvay*.pgm; do
	avg_total=0
	avg_img_load=0
	avg_gpu_init=0
	avg_det_time=0
	avg_total_less_gpu=0
	i=0
	while [ $i -lt $iterations ]; do
		times=($(../vj "$scale" output.pgm | tail -n 10))
		avg_total=$(bc <<< 'scale=4; $avg_total + $times[1]')
		avg_det_time=$(bc <<< 'scale=4; $avg_det_time + $times[3]')
		avg_img_load=$(bc <<< 'scale=4; $avg_img_load + $times[5]')
		avg_gpu_init=$(bc <<< 'scale=4; $avg_gpu_init + $times[7]')
		avg_total_less_gpu=$(bc <<< 'scale=4; $avg_total_less_gpu + $times[9]')
		i=$((i+1))
	done
	echo $scale
	echo "average total: $avg_total"
	echo "average image load: $avg_img_load"
	echo "average GPU init: $avg_gpu_init"
	echo "average detection time: $avg_det_time"
	echo "average total minus GPU init: $avg_total_less_gpu"
done
