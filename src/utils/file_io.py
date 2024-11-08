# src/utils/file_io.py
import pandas as pd
import h5py
import os

def import_from_hdf5(filename, key='ccr_data'):
    """
    Import data from an HDF5 file into a pandas DataFrame.

    Args:
        filename (str): Path to the HDF5 file.
        key (str): The key within the HDF5 file to load the data from (default is 'ccr_data').

    Returns:
        pd.DataFrame: The DataFrame containing the imported data.
    """
    ccr_data_index = pd.read_hdf(filename, key=key)
    print(f"DataFrame imported from {filename}.")
    return ccr_data_index

def export_to_hdf5(ccr_data_index, filename, key='ccr_data'):
    """
    Export a pandas DataFrame to an HDF5 file.

    Args:
        ccr_data_index (pd.DataFrame): The DataFrame to export.
        filename (str): Path to the HDF5 file where the data should be saved.
        key (str): The key under which the data will be stored in the HDF5 file (default is 'ccr_data').

    Returns:
        None
    """
    ccr_data_index.to_hdf(filename, key=key, mode='w')
    print(f"DataFrame exported to {filename}.")