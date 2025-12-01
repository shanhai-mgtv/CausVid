PIDS=$(ps aux | grep python | grep -v grep | awk '{print $2}')

for PID in $PIDS; do
	    # echo "Killing Python process with PID: $PID"
	    kill -9 $PID
	done

	echo "All Python processes have been terminated."


script_path="/mnt/cfs/shanhai/jyutong/hyc/causvid/distillation_wan22.sh"
log_dir="/mnt/cfs/shanhai/jyutong/hyc/causvid/logs"
log_path="$log_dir/$(date +%Y%m%d_%H%M%S).log"
ips=(self worker14 worker15 worker18)
nodes=${#ips[@]}

echo "ALL Worker IP:"
printf "  %s\n" "${ips[@]}"

for idx in "${!ips[@]}"; do
    ip="${ips[idx]}"
    echo "Processing $ip... ${idx}/${nodes} master node is ${ips[0]}"
    ssh $ip "bash $script_path $nodes $idx" ${log_path} &
done

while [ ! -f "$log_path" ]; do
    echo "Waiting conda envs activation..."
    sleep 1
done