import os
import sys
from utils.file_io import import_from_hdf5, export_to_hdf5
from get_footprint import full_output

def run_cli():
    """
    Command-line interface (CLI) for running the full data processing pipeline.
    
    This function prompts the user for input values such as file paths, ground truth numbers, 
    region names, and footprint range, and then calls the `full_output` function to process 
    the data accordingly. The function handles both full data processing as well as importing 
    previously selected data indices, depending on user input.

    The user can choose whether to:
    1. Process the data from scratch and perform footprint selection.
    2. Import previously selected indices from an HDF5 file for further processing.

    After processing, the results are generated, visualized, and saved to a uniquely named output directory.
    
    Functionality:
        - Prompts the user for necessary inputs.
        - Calls the `full_output` function to run data processing, including footprint calculations.
        - Handles errors during the processing and outputs appropriate messages.

    Exceptions:
        - If there is an error during processing, it will display the error message to the user.

    Returns:
        None
    """
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
