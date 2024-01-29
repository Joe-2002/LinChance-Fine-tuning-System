import time
import streamlit as st
import os
import subprocess
# Create global variables to store the process and output pipe
process = None
out_r, out_w = None, None

@st.cache_resource
def get_pipe():
    print('Creating pipe')
    global out_r, out_w
    out_r, out_w = os.pipe()
    return out_r, out_w

@st.cache_resource
def get_Popen(out_w,command):
    print('Creating process')
    global process
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=out_w, 
        stderr=out_w, 
        universal_newlines=False
    )
    return process

def start(command):
    global process, out_r, out_w

    if process is None:
        out_r, out_w = get_pipe()
        process = get_Popen(out_w,command)

        st.markdown("## Output")
        output_container = st.empty()

        stop_button = st.button("Turn to the background")

        while True:
            if stop_button:
                process.terminate()
                st.warning("Process terminated.")
                process = None  # Reset process
                break

            raw_data = os.read(out_r, 1000)
            try:
                logs = raw_data.decode("utf-8", errors="ignore")
                output_container.text(logs)
            except UnicodeDecodeError as e:
                print(f'UnicodeDecodeError: {e}')
            time.sleep(0.5)

    # # Close the write end of the pipe
    # os.close(out_w)
    # os.close(out_r)

