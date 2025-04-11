pkill python
git pull

CHECKPOINT_MODE=$1
device=$2
world_size=$3
h_failure_probability=$4
config=$5
for ((i=0; i<$world_size; i=i+1))
do
    touch "out$i.txt"
    (sleep 1; python -u "convergence_training.py" $CHECKPOINT_MODE $device $i $world_size $h_failure_probability $config>"out$i.txt")&


done