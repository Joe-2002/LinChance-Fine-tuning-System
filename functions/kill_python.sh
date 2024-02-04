#!/bin/bash

# Find all Python processes associated with NVIDIA devices
nvidia_python_pids=$(fuser -v /dev/nvidia* 2>/dev/null | grep " python" | awk '{print $2}')

# Add Streamlit processes to the list
streamlit_pids=$(pgrep -f "streamlit")

# Combine the PIDs from both lists
all_pids="$nvidia_python_pids $streamlit_pids"

# Loop through and kill each process
for pid in $all_pids; do
    echo "Killing process with PID: $pid"
    kill -9 $pid
done

echo "Script execution completed."
