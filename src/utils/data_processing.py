# src/utils/data_processing.py
import numpy as np
import pandas as pd
import scipy
import pyproj
import pathlib
from scipy.io import loadmat
from pyproj import Transformer, Proj
from pathlib import Path as PathlibPath
import h5py

# Load and process lidar data
def load_lidar_data(lidar_mat_file):
    """
    Load lidar data from a MAT-file.

    Args:
        lidar_mat_file (str): The path to the MAT-file containing lidar data.

    Returns:
        dict or None: Loaded MAT-file data if the file exists, otherwise None.
    """
    file_path = PathlibPath(lidar_mat_file)
    return scipy.io.loadmat(lidar_mat_file) if file_path.is_file() else None

# Read data from the HDF5 file
def read_and_transform_data(h5_file_path, gt_num, lidar_mat_file):
    """
    Read ground truth data from an HDF5 file and transform the coordinates from 
    latitude/longitude to UTM (Universal Transverse Mercator).

    Args:
        h5_file_path (str): Path to the HDF5 file containing ground truth data.
        gt_num (str): Ground truth dataset identifier (group name in the HDF5 file).
        lidar_mat_file (str): Path to the MAT-file containing lidar data for UTM correction.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Corrected ground truth data with UTM coordinates and altitude.
            - pd.DataFrame: Original ground truth data with uncorrected UTM coordinates.
            - list: UTM corrections [easting correction, northing correction, vertical correction].
    """
    mat_data = load_lidar_data(lidar_mat_file)
    utm_correction = [0, 0, 0] if not mat_data else [eastingCorrection, northingCorrection, verticalCorrection]

    with h5py.File(h5_file_path, 'r') as file:
        lats = np.array(file[f'/{gt_num}/heights/lat_ph'])
        lons = np.array(file[f'/{gt_num}/heights/lon_ph'])
        z = np.array(file[f'/{gt_num}/heights/h_ph'])

    track_direction = 'Descending' if lats[0] >= lats[-1] else 'Ascending'

    transformer = Transformer.from_proj(
        Proj(proj='latlong', datum='WGS84'),
        Proj(proj='utm', zone=13, datum='WGS84', hemisphere='N')
    )
    
    utme_uncorrected, utmn_uncorrected = transformer.transform(lons, lats)
    utme_corrected = utme_uncorrected + utm_correction[0]
    utmn_corrected = utmn_uncorrected + utm_correction[1]
    alt_corrected = z + utm_correction[2]

    gt_data_corrected = pd.DataFrame({
        'UTM Easting': utme_corrected,
        'UTM Northing': utmn_corrected,
        'Altitude': alt_corrected
    })
    
    gt_data = pd.DataFrame({
        'gt_x': utme_uncorrected,
        'gt_y': utmn_uncorrected,
        'gt_z': z,
        'gt_trackDirection': track_direction
    })
    return gt_data_corrected, gt_data, utm_correction

# Load CCR truth data
def load_ccr_truth_data(region_name):
    """
    Load the CCR truth data (cloud calibration reference data) for a specified region.

    Args:
        region_name (str): The region name, either 'wsmr' or 'antarctic'.

    Returns:
        pd.DataFrame: A DataFrame containing the CCR truth data including coordinates, names, and relative heights.
    """
    current_directory = PathlibPath(__file__).parent
    file_name = "wsmr_cc_locations_new.mat" if region_name.lower() == 'wsmr' else "antarctic_cc_locations_new.mat"
    mat_file_path = current_directory / "supportFiles" / file_name

    ccr_truth_data = scipy.io.loadmat(mat_file_path)
    
    # Initialize variables
    xlabelStr, ylabelStr = ('UTM Easting (m)', 'UTM Northing (m)') if region_name.lower() == 'wsmr' else ('Polar Stereo X (m)', 'Polar Stereo Y (m)')
    
    if region_name.lower() == 'wsmr':
        ccrX, ccrY = ccr_truth_data['ccrX'].flatten(), ccr_truth_data['ccrY'].flatten()
        ccrNames = [str(name) for name in ccr_truth_data['ccrNames'].ravel()]
        ccrRelativeHeights = ccr_truth_data['ccrRelativeHeights'].flatten()
    else:
        ccr_heights = ccr_truth_data['ccrStruct']['height_m'].flatten()
        valid_ccrs = ccr_heights != 0
        ccrX = ccr_truth_data['ccrX'].flatten()[valid_ccrs]
        ccrY = ccr_truth_data['ccrY'].flatten()[valid_ccrs]
        ccrNames = [str(name) for name in ccr_truth_data['ccrNames'].ravel()[valid_ccrs]]
        ccrRelativeHeights = ccr_truth_data['ccrRelativeHeights'].flatten()

    return pd.DataFrame({
        'ccrX': ccrX,
        'ccrY': ccrY,
        'ccrNames': ccrNames,
        'xlabelStr': xlabelStr,
        'ylabelStr': ylabelStr,
        'ccrRelativeHeights': ccrRelativeHeights,
    })

def get_interp_x(distPts, y2, e2_thresh):
    """
    Perform linear interpolation to find the x-values corresponding to a specific y-value threshold.

    Args:
        distPts (array-like): Array of x-values (distances).
        y2 (array-like): Array of y-values.
        e2_thresh (float): The threshold value of y to interpolate for.

    Returns:
        np.ndarray: The interpolated x-values corresponding to the e2_thresh.
    """
    # Combine x and y values into a single array and sort by x values
    array_in = np.column_stack((distPts, y2))
    array_in_sorted = array_in[array_in[:, 0].argsort()]
    
    x_vals = array_in_sorted[:, 0]
    y_vals = array_in_sorted[:, 1]
    
    x_vals_interp = []

    for i in range(len(y_vals) - 1):
        x_val_curr = x_vals[i]
        x_val_post = x_vals[i + 1]
        y_val_curr = y_vals[i]
        y_val_post = y_vals[i + 1]
        
        # Check if e2_thresh is between y_val_curr and y_val_post
        if (y_val_curr <= e2_thresh <= y_val_post) or (y_val_post <= e2_thresh <= y_val_curr):
            # Linear interpolation formula
            x_interp = x_val_curr + (x_val_post - x_val_curr) * ((e2_thresh - y_val_curr) / (y_val_post - y_val_curr))
            x_vals_interp.append(x_interp)
    
    return np.array(x_vals_interp)