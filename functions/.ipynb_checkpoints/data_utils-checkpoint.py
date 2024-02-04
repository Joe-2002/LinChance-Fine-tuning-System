# data_utils.py

import os
import json
import streamlit as st

def upload_and_display_data(data_dir):
    uploaded_file = st.sidebar.file_uploader("Select your dataset and upload it")

    # Automatically name the uploaded dataset as "lima.json"
    file_name = "lima.json"

    if uploaded_file is not None:
        save_file_path = os.path.join(data_dir, file_name)

        # Save the uploaded file as "lima.json"
        with open(save_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.sidebar.success(f"Success: {save_file_path}")
    
    return data_dir

def select_and_display_dataset(data_dir):
    file_list = [file for file in os.listdir(data_dir) if file.endswith(".json")]

    if not file_list:
        st.warning("No JSON files found in the data directory.")
    else:
        selected_dataset = st.sidebar.selectbox(
            "Choose your dataset",
            file_list,
            index=0  # Set the default index to 0
        )

        if selected_dataset:
            st.title("Dataset")
            st.caption(f'Selected Dataset: :green[{selected_dataset}]')
            selected_file_path = os.path.join(data_dir, selected_dataset)

            with open(selected_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            first_five_data = data[:5]

            expander = st.expander("View first five data entries", expanded=False)
            with expander:
                for entry in first_five_data:
                    st.code(entry)
