PIDS=$(ps aux | grep python | grep -v grep | awk '{print $2}')

for PID in $PIDS; do
	    # echo "Killing Python process with PID: $PID"
	    kill -9 $PID
	done

	echo "All Python processes have been terminated."


script_path="/mnt/shanhai-ai/shanhai-workspace/jyutong/hyc/CausVid/generate_ode.sh"
log_dir="/mnt/shanhai-ai/shanhai-workspace/jyutong/hyc/CausVid/ode_logs"
log_path="$log_dir/$(date +%Y%m%d_%H%M%S).log"
ips=(8A-0 8A-2 8A-3 8A-4)
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