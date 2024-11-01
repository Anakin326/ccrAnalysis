import os
import sys
from utils.file_io import import_from_hdf5, export_to_hdf5
from get_footprint import full_output

def run_cli():
    # Get the H5 file path from the user
    h5_file_path = input("Enter the path to the H5 file: ")
    
    # Get the Ground Truth Number from the user
    gt_num = input("Enter the Ground Truth Number: ")
    
    # Get the Region Name from the user
    region_name = input("Enter the Region Name (WSMR/Antarctic): ").lower()
    
    # Get the Footprint Range from the user
    footprint_range = input("Enter the Footprint Range (default is '5:0.1:20'): ")
    if not footprint_range:  # Set default if empty
        footprint_range = '5:0.1:20'
    
    # Ask if they want to import indices
    import_indices = input("Do you want to import indices? (yes/no): ").strip().lower()
    
    imported_h5 = None
    if import_indices == 'yes':
        imported_h5 = input("Enter the path to the HDF5 file containing indices: ")

    try:
        # Call the full_output function with the collected inputs
        full_output(h5_file_path, gt_num, region_name, footprint_range, imported_h5=imported_h5, run_select_data=(imported_h5 is None))
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
