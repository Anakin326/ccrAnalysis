# src/utils/file_io.py
import pandas as pd
import h5py
import os

def import_from_hdf5(filename, key='ccr_data'):
    ccr_data_index = pd.read_hdf(filename, key=key)
    print(f"DataFrame imported from {filename}.")
    return ccr_data_index

def export_to_hdf5(ccr_data_index, filename, key='ccr_data'):
    ccr_data_index.to_hdf(filename, key=key, mode='w')
    print(f"DataFrame exported to {filename}.")