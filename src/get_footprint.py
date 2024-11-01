# Standard library imports
import os
import time
import itertools
# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy import stats
from scipy.stats import norm
from scipy.interpolate import interp1d
# Local application/library imports
from utils.data_processing import load_lidar_data, read_and_transform_data, load_ccr_truth_data, get_interp_x
from utils.plotting import SelectFromCollection, makeCircle, plot_selection_data, plot_figures, find_ccr_regions
from utils.file_io import import_from_hdf5, export_to_hdf5

# Handle user selections
def handle_selections(fig, ax1, scatter_plot, gt_data_corrected, ccr_truth_data):
    selector = SelectFromCollection(ax1, scatter_plot)
    all_selected_points = []
    current_selection_type = 0
    mean_CCR = None
    mean_Ground = None
    ccrData = None
    groundData = None
    closest_ccr_name = None
    closest_ccr_height = None
    closest_ccr_index = None
    heightDelta = None
    closest_ccr_x = None
    closest_ccr_y = None
    data_list = []

    def accept(event):
        nonlocal current_selection_type, mean_CCR, mean_Ground, ccrData, groundData, closest_ccr_name, closest_ccr_height, heightDelta, closest_ccr_x, closest_ccr_y

        if event.key == "enter":
            selected_points = selector.xys[selector.ind]
            if selected_points.size == 0:
                return

            selected_utm_northings = selected_points[:, 0]
            matching_rows = gt_data_corrected[gt_data_corrected['UTM Northing'].isin(selected_utm_northings)]
            mean_utm_easting = matching_rows['UTM Easting'].mean()
            mean_utm_northing = matching_rows['UTM Northing'].mean()

            if current_selection_type == 0:  # CCR points
                mean_CCR = selected_points[:, 1].mean()
                ccrData = selected_points
                all_selected_points.append(('CCR', selected_points))
                print(f"CCR Selection {len(all_selected_points) // 2 + 1}: Mean Altitude: {mean_CCR}")

            elif current_selection_type == 1:  # Ground points
                mean_Ground = selected_points[:, 1].mean()
                groundData = selected_points
                all_selected_points.append(('Ground', selected_points))
                print(f"Ground Selection {len(all_selected_points) // 2 + 1}: Mean Altitude: {mean_Ground}")

                if all_selected_points and 'CCR' in all_selected_points[-2][0]:  # Ensure previous selection was a CCR
                    CCR_Height = mean_CCR - mean_Ground
                    ccrHeight_ft = CCR_Height * 3.28084
                    # Calculate distances to CCRs using the DataFrame
                    distances_to_ccrs = np.sqrt((ccr_truth_data['ccrX'].values - mean_utm_easting) ** 2 +
                                                (ccr_truth_data['ccrY'].values - mean_utm_northing) ** 2)
                    closest_ccr_index = np.argmin(distances_to_ccrs)
                    closest_ccr_name = ccr_truth_data['ccrNames'].iloc[closest_ccr_index]
                    closest_ccr_height = ccr_truth_data['ccrRelativeHeights'].iloc[closest_ccr_index]
                    heightDelta = ccrHeight_ft - closest_ccr_height
                    heightDeltaSign = "+" if heightDelta >= 0 else "-"
                    closest_ccr_x = ccr_truth_data['ccrX'].iloc[closest_ccr_index]
                    closest_ccr_y = ccr_truth_data['ccrY'].iloc[closest_ccr_index]
                        
                    try:
                        ccNum = int(closest_ccr_name.strip("[]'")[3:])  # Extract number from 'CCR123'
                        print(f"Closest CCR: {closest_ccr_name}")
                        print(f"Truth Height for {closest_ccr_name}: {closest_ccr_height}")
                        print(f"Height Delta: {heightDeltaSign}{heightDelta}")

                        if isinstance(ccrData, np.ma.MaskedArray):
                            ccrData = ccrData.filled(np.nan)  # Fill with NaN or another value if desired
                        if isinstance(groundData, np.ma.MaskedArray):
                            groundData = groundData.filled(np.nan)  # Similarly convert groundData

                        # Store the CCR and ground data for this ccNum
                        data_list.append({
                            'ccrData': ccrData,
                            'groundData': groundData,
                            'closest_ccr_name': closest_ccr_name,
                            'closest_ccr_height': closest_ccr_height,
                            'heightDelta': heightDelta,
                            'closest_ccr_index': closest_ccr_index,
                            'closest_ccr_x': closest_ccr_x,
                            'closest_ccr_y': closest_ccr_y,
                            'ccNum': ccNum,
                        })
                        print(f"Data stored for CCR number {ccNum}")
                    except ValueError:
                        print("Error extracting CCR number.")

            selector.disconnect()
            ax1.set_title("Select CCR points, then ground points.")
            current_selection_type = 1 - current_selection_type  # Toggle selection type
            selector.__init__(ax1, scatter_plot)
            fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", accept)
    plt.show()
    ccr_data_index = pd.DataFrame(data_list)  # Create DataFrame after accepting selections

    return ccr_data_index

def get_offset(ccr_data_index, gt_data, r_nominal, ccr_truth_data, utm_correction):
    cc_Struct_dict = {}
    measData_x = gt_data['gt_x'].values
    measData_y = gt_data['gt_y'].values
    measData_z = gt_data['gt_z'].values
    sigma = 2
    ccrX_truthRot_dict = {}
    ccrY_truthRot_dict = {}
    plot_data_list = {}

    def rotate_data(data, rotation_point, theta_rad):
        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], 
                      [np.sin(theta_rad), np.cos(theta_rad)]])
        return R @ (data - rotation_point[:, None])

    # Populate cc_Struct_dict based on ccr_data_index
    for current_index, row in ccr_data_index.iterrows():
        # Extract the necessary data from the DataFrame row
        ccrData = row['ccrData']
        closest_ccr_index = row['closest_ccr_index']
        closest_ccr_x = row['closest_ccr_x']
        closest_ccr_y = row['closest_ccr_y']
        ccNum = row['ccNum']
        
        # Get linear fit and rotation parameters
        gt2PolyFit = np.polyfit(measData_x, measData_y, 1)
        slope, yInt = gt2PolyFit[0], gt2PolyFit[1]
        x1, x2 = measData_x[0], measData_x[-1]
        y1, y2 = slope * x1 + yInt, slope * x2 + yInt
        deltaX, deltaY = x2 - x1, y2 - y1
        phi = np.degrees(np.arctan2(deltaY, deltaX))

        # Translate GT2R line to (0,0) and Rotate GT2 strong beam line to be vertical
        yRotPt = measData_y[0]
        xRotPt = (yRotPt - yInt) / slope
        theta_rad = np.radians(90 - phi)
        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])
        
        # Rotate measurements and CCR data
        measData_rotated = rotate_data(np.vstack([measData_x, measData_y]), np.array([xRotPt, yRotPt]), theta_rad)
        ccr_truth_rot = rotate_data(np.vstack([closest_ccr_x, closest_ccr_y]), np.array([xRotPt, yRotPt]), theta_rad)

        ccrX_truthRot_dict[current_index] = ccr_truth_rot[0]
        ccrY_truthRot_dict[current_index] = ccr_truth_rot[1]

        #Rotate all CCR points
        ccr_rot = R @ np.vstack([ccr_truth_data['ccrX'].values - xRotPt, ccr_truth_data['ccrY'].values - yRotPt])
        ccrX_Rot, ccrY_Rot = ccr_rot[0, :], ccr_rot[1, :]

        ccArray = np.empty((0, 2))
        ccArray = np.vstack((ccArray, ccrData))
        
        cc_Struct, plot_data = getCCRdata(ccNum, measData_x, measData_y, ccArray, xRotPt, yRotPt, R, closest_ccr_x, closest_ccr_y, r_nominal, sigma)
        plot_data["ccNum"] = ccNum
        cc_Struct_dict[current_index] = cc_Struct
        plot_data_list[current_index] = plot_data

    centroidSides = ['Right', 'Left']
    combo_array, combo_arrayPred = get_combos(cc_Struct_dict, centroidSides, ccrX_truthRot_dict, ccrY_truthRot_dict)

    def get_min_combo(array):
        min_row = np.argmin([row[44] for row in array])
        return array[min_row][:6], array[min_row][18], array[min_row][19]

    min_combo_pred, min_x_shift_pred, min_y_shift_pred = get_min_combo(combo_arrayPred)
    min_combo, min_x_shift, min_y_shift = get_min_combo(combo_array)

    # Rotate mean offsets to Easting/Northing frame
    def compute_shift(x_shift, y_shift):
        return np.linalg.solve(np.array([[np.cos(theta_rad), -np.sin(theta_rad)], 
                                          [np.sin(theta_rad), np.cos(theta_rad)]]), 
                                       np.array([x_shift, y_shift]))

    e_shift, n_shift = compute_shift(min_x_shift, min_y_shift)
    e_shift_pred, n_shift_pred = compute_shift(min_x_shift_pred, min_y_shift_pred)

    # Get centroid error RMSE values
    min_row = np.argmin([row[44] for row in combo_array])
    min_row_pred = np.argmin([row[44] for row in combo_arrayPred])
    x_centroid_error_rmse = combo_array[min_row][44]
    y_centroid_error_rmse = combo_array[min_row][45]
    x_centroid_error_pred_rmse = combo_arrayPred[min_row_pred][44]
    y_centroid_error_pred_rmse = combo_arrayPred[min_row_pred][45]

    results = np.array([
    2 * r_nominal,         # Diameter (nominal)
    e_shift_pred,         # Predicted Easting shift
    n_shift_pred,         # Predicted Northing shift
    min_x_shift_pred,         # Predicted Cross-Track shift
    min_y_shift_pred,         # Predicted Along-Track shift
    x_centroid_error_pred_rmse,  # RMSE for Cross-Track error prediction
    y_centroid_error_pred_rmse   # RMSE for Along-Track error prediction
    ])
    
    # Initialize direction list
    direction = []
    for combo in min_combo_pred:
        if combo is not None:  # Check if combo is not None
            str_splits = combo.split('*')  # Splitting the string by '*'
            direction.append(str_splits[1])  # Store the second part (direction)
        else:
            direction.append(None)  # Append None or handle as needed

    # Print output
    print(f'\nMeasured Cross-Track Offset: {min_x_shift:.2f} m (RMSE: {x_centroid_error_rmse:.2f} m)')
    print(f'Measured Along-Track Offset: {min_y_shift:.2f} m (RMSE: {y_centroid_error_rmse:.2f} m)\n')
    print(f'Statistical Cross-Track Offset: {min_x_shift_pred:.2f} m (RMSE: {x_centroid_error_pred_rmse:.2f} m)')
    print(f'Statistical Along-Track Offset: {min_y_shift_pred:.2f} m (RMSE: {y_centroid_error_pred_rmse:.2f} m)\n')
    print(f'Measured Easting Offset: {e_shift:.2f} m')
    print(f'Measured Northing Offset: {n_shift:.2f} m\n')
    print(f'Statistical Easting Offset: {e_shift_pred:.2f} m')
    print(f'Statistical Northing Offset: {n_shift_pred:.2f} m\n')

    return results, combo_arrayPred, cc_Struct_dict, direction, plot_data_list

    ### GET CCRDATA CALL ###

def getCCRdata(ccNum, measData_x, measData_y, ccArray, xRotPt, yRotPt, R, ccrX_truth, ccrY_truth, r_nominal, sigma):
    inds = np.isin(measData_y, ccArray[:,0])
    ccX = measData_x[inds]
    ccY = measData_y[inds]

    # Perform rotation assuming R is a 2x2 rotation matrix
    ccXY = np.vstack([ccX - xRotPt, ccY - yRotPt])
    ccXYrot = R @ ccXY
    ccXrot, ccYrot = ccXYrot[0, :], ccXYrot[1, :]

    # Rotate CCR Truth data
    ccXYrot_truth = R @ np.vstack([(ccrX_truth - xRotPt), (ccrY_truth - yRotPt)])
    ccXrot_truth, ccYrot_truth = ccXYrot_truth[0], ccXYrot_truth[1, :]
    
    # Fit normal distribution to the rotated data
    histMean, histStd = np.mean(ccYrot), np.std(ccYrot)
    yMinActualRot, yMaxActualRot = np.min(ccYrot), np.max(ccYrot)

    # Generate data points for Gaussian fit
    distPts = histMean + sigma * histStd * np.random.randn(1000)
    y2 = norm.pdf(distPts, loc=histMean, scale=histStd)

    # Calculate predicted bounds
    gaussDist = sigma * histStd
    yMinPredictedRot = histMean - gaussDist
    yMaxPredictedRot = histMean + gaussDist
    yMinPredictedRot3 = histMean - 3 * histStd
    yMaxPredictedRot3 = histMean + 3 * histStd

    # Get percent of Gaussian in actual data range
    cDist = norm.cdf(distPts, loc=histMean, scale=histStd)
    interp_func = interp1d(distPts, cDist, bounds_error=False, fill_value=(0, 1))
    
    pctLoBound = interp_func(yMinActualRot)
    pctHiBound = interp_func(yMaxActualRot)
    pctTotBound = (pctHiBound - pctLoBound) * 100

    pctLoBoundPred = interp_func(yMinPredictedRot)
    pctHiBoundPred = interp_func(yMaxPredictedRot)
    pctTotBoundPred = (pctHiBoundPred - pctLoBoundPred) * 100

    # Get sigma of Gaussian in actual data range
    actualMinSigma = abs(yMinActualRot - histMean) / histStd
    actualMaxSigma = abs(yMaxActualRot - histMean) / histStd
    actualTotSigma = actualMinSigma + actualMaxSigma
    
    # Interpolate to get more accurate crossing points
    e2_thresh = 0.135 * np.max(y2)
    e2_range = get_interp_x(distPts, y2, e2_thresh)
    e2_minX, e2_maxX = e2_range[0], e2_range[1]

    plot_data = {
        "ccYrot": ccYrot,
        "distPts": distPts,
        "y2": y2,
        "histMean": histMean,
        "histStd": histStd,
        "yMinPredictedRot": yMinPredictedRot,
        "yMaxPredictedRot": yMaxPredictedRot,
        "yMinActualRot": yMinActualRot,
        "yMaxActualRot": yMaxActualRot,
        "e2_minX": e2_minX,
        "e2_maxX": e2_maxX,
        "e2_thresh": e2_thresh,
        "sigma": sigma,
        "pctTotBoundPred": pctTotBoundPred,
        "actualTotSigma": actualTotSigma,
        "pctTotBound": pctTotBound
    }

    r_3sig = r_nominal
    r_actual = r_nominal

    ###MEASURED APPROACH ###
    # Get actuals in CT/AT frame
    cc_Struct = {
        'x': ccX,
        'y': ccY,
        'xRot': ccXrot,
        'yRot': ccYrot,
        'chordNumPts': f" ({len(ccXrot):.0f} pts)"
    }
    
    y_rot_max, y_rot_min = np.max(ccYrot), np.min(ccYrot)
    cc_Struct['chord'] = abs(y_rot_max - y_rot_min)
    y_half_chord = cc_Struct['chord'] / 2
    x_half_chord = np.nan_to_num(np.sqrt(r_actual**2 - y_half_chord**2))

    # Calculate center and right/left coordinates in rotated frame
    x_rot_mean = np.mean(ccXrot)
    y_rot_avg = (y_rot_max + y_rot_min) / 2
    cc_Struct.update({
        'xChordCenterRot': x_rot_mean,
        'yChordCenterRot': y_rot_avg,
        'xCenterRightRot': x_rot_mean + x_half_chord,
        'yCenterRightRot': y_rot_max - y_half_chord,
        'xCenterLeftRot': x_rot_mean - x_half_chord,
        'yCenterLeftRot': y_rot_max - y_half_chord
    })

    # Generate circles
    cc_Struct['xCircleRightRot'], cc_Struct['yCircleRightRot'] = makeCircle(cc_Struct['xCenterRightRot'], cc_Struct['yCenterRightRot'], r_actual)
    cc_Struct['xCircleLeftRot'], cc_Struct['yCircleLeftRot'] = makeCircle(cc_Struct['xCenterLeftRot'], cc_Struct['yCenterLeftRot'], r_actual)

    # Apply rotation matrix and offsets for X/Y frame
    def apply_rotation_offset(x_rot, y_rot):
        x_y_rot = np.linalg.solve(R, [x_rot, y_rot])
        return x_y_rot[0] + xRotPt, x_y_rot[1] + yRotPt

    cc_Struct['xChordCenter'], cc_Struct['yChordCenter'] = apply_rotation_offset(cc_Struct['xChordCenterRot'], cc_Struct['yChordCenterRot'])
    cc_Struct['xCenterRight'], cc_Struct['yCenterRight'] = apply_rotation_offset(cc_Struct['xCenterRightRot'], cc_Struct['yCenterRightRot'])
    cc_Struct['xCircleRight'], cc_Struct['yCircleRight'] = apply_rotation_offset(cc_Struct['xCircleRightRot'], cc_Struct['yCircleRightRot'])
    cc_Struct['xCenterLeft'], cc_Struct['yCenterLeft'] = apply_rotation_offset(cc_Struct['xCenterLeftRot'], cc_Struct['yCenterLeftRot'])
    cc_Struct['xCircleLeft'], cc_Struct['yCircleLeft'] = apply_rotation_offset(cc_Struct['xCircleLeftRot'], cc_Struct['yCircleLeftRot'])

    ## STATISTICAL APPROACH ##
    yRotPred = np.linspace(yMinPredictedRot, yMaxPredictedRot, 20)
    cc_Struct.update({
        'xRotPred': np.full_like(yRotPred, x_rot_mean),
        'yRotPred': yRotPred,
        'chordPred': abs(np.ptp(yRotPred)),
    })

    y_half_chord_pred = cc_Struct['chordPred'] / 2
    x_half_chord_pred = np.nan_to_num(np.sqrt(r_3sig**2 - y_half_chord_pred**2))

    # Centers for predicted values
    cc_Struct.update({
        'xChordCenterRotPred': x_rot_mean,
        'yChordCenterRotPred': np.mean(yRotPred),
        'xCenterRightRotPred': x_rot_mean + x_half_chord_pred,
        'yCenterRightRotPred': np.max(yRotPred) - y_half_chord_pred,
        'xCenterLeftRotPred': x_rot_mean - x_half_chord_pred,
        'yCenterLeftRotPred': np.max(yRotPred) - y_half_chord_pred,
    })

    # Generate Circles
    cc_Struct['xCircleRightRotPred'], cc_Struct['yCircleRightRotPred'] = makeCircle(cc_Struct['xCenterRightRotPred'], cc_Struct['yCenterRightRotPred'], r_3sig)
    cc_Struct['xCircleLeftRotPred'], cc_Struct['yCircleLeftRotPred'] = makeCircle(cc_Struct['xCenterLeftRotPred'], cc_Struct['yCenterLeftRotPred'], r_3sig)

    # Get predicted values in CT/AT frame (3 sigma)
    yRotPred3 = np.linspace(yMinPredictedRot3, yMaxPredictedRot3, 20)
    cc_Struct.update({
        'xRotPred3': np.full_like(yRotPred3, x_rot_mean),
        'yRotPred3': yRotPred3,
        'chordPred3': abs(np.ptp(yRotPred3)),
    })

    # Get predicted values in X/Y frame
    xyPred = np.linalg.solve(R, np.vstack([cc_Struct['xRotPred'], cc_Struct['yRotPred']]))
    cc_Struct['xPred'] = xyPred[0, :] + xRotPt
    cc_Struct['yPred'] = xyPred[1, :] + yRotPt

    xyChordCenterPred = np.linalg.solve(R, np.array([cc_Struct['xChordCenterRotPred'], cc_Struct['yChordCenterRotPred']]))
    cc_Struct['xChordCenterPred'] = xyChordCenterPred[0] + xRotPt
    cc_Struct['yChordCenterPred'] = xyChordCenterPred[1] + yRotPt

    xyCenterRightPred = np.linalg.solve(R, np.array([cc_Struct['xCenterRightRotPred'], cc_Struct['yCenterRightRotPred']]))
    cc_Struct['xCenterRightPred'] = xyCenterRightPred[0] + xRotPt
    cc_Struct['yCenterRightPred'] = xyCenterRightPred[1] + yRotPt

    xyCenterLeftPred = np.linalg.solve(R, np.array([cc_Struct['xCenterLeftRotPred'], cc_Struct['yCenterLeftRotPred']]))
    cc_Struct['xCenterLeftPred'] = xyCenterLeftPred[0] + xRotPt
    cc_Struct['yCenterLeftPred'] = xyCenterLeftPred[1] + yRotPt

    xyCircleRightPred = np.linalg.solve(R, np.vstack([cc_Struct['xCircleRightRotPred'], cc_Struct['yCircleRightRotPred']]))
    cc_Struct['xCircleRightPred'] = xyCircleRightPred[0] + xRotPt
    cc_Struct['yCircleRightPred'] = xyCircleRightPred[1] + yRotPt

    xyCircleLeftPred = np.linalg.solve(R, np.vstack([cc_Struct['xCircleLeftRotPred'], cc_Struct['yCircleLeftRotPred']]))
    cc_Struct['xCircleLeftPred'] = xyCircleLeftPred[0] + xRotPt
    cc_Struct['yCircleLeftPred'] = xyCircleLeftPred[1] + yRotPt

    # Make circles around CCR data
    cc_Struct['ccrCircleXRot'], cc_Struct['ccrCircleYRot'] = makeCircle(ccXrot_truth, ccYrot_truth, r_3sig)
    ccrCircleXY_input = np.vstack((cc_Struct['ccrCircleXRot'], cc_Struct['ccrCircleYRot']))
    ccrCircleXY = np.linalg.solve(R, ccrCircleXY_input)

    cc_Struct.update({
        'ccrCircleX': ccrCircleXY[0] + xRotPt,
        'ccrCircleY': ccrCircleXY[1] + yRotPt,
    })

    ## OUTPUT ##
    # Define centroids and their predictions
    cc_Struct['xCentroidDeltaLeft'] = ccXrot_truth - cc_Struct['xCenterLeftRot']
    cc_Struct['yCentroidDeltaLeft'] = ccYrot_truth - cc_Struct['yCenterLeftRot']
    cc_Struct['xCentroidDeltaLeftPred'] = ccXrot_truth - cc_Struct['xCenterLeftRotPred']
    cc_Struct['yCentroidDeltaLeftPred'] = ccYrot_truth - cc_Struct['yCenterLeftRotPred']

    # Calculate RMSE
    measX_rmse = np.sqrt(np.mean((ccXrot_truth - cc_Struct['xRot'])**2))
    measY_rmse = np.sqrt(np.mean((ccYrot_truth - cc_Struct['yRot'])**2))
    statX_rmse = np.sqrt(np.mean((ccXrot_truth - cc_Struct['xRotPred'])**2))
    statY_rmse = np.sqrt(np.mean((ccYrot_truth - cc_Struct['yRotPred'])**2))

    # Define right centroids and their predictions
    cc_Struct['xCentroidDeltaRight'] = ccXrot_truth - cc_Struct['xCenterRightRot']
    cc_Struct['yCentroidDeltaRight'] = ccYrot_truth - cc_Struct['yCenterRightRot']
    cc_Struct['xCentroidDeltaRightPred'] = ccXrot_truth - cc_Struct['xCenterRightRotPred']
    cc_Struct['yCentroidDeltaRightPred'] = ccYrot_truth - cc_Struct['yCenterRightRotPred']

    #Final Structure Updates
    cc_Struct.update({
        'measX_delta': ccXrot_truth - cc_Struct['xChordCenterRot'],
        'measY_delta': ccYrot_truth - cc_Struct['yChordCenterRot'],
        'statX_delta': ccXrot_truth - cc_Struct['xChordCenterRotPred'],
        'statY_delta': ccYrot_truth - cc_Struct['yChordCenterRotPred'],
        'measX_rmse': measX_rmse,
        'measY_rmse': measY_rmse,
        'statX_rmse': statX_rmse,
        'statY_rmse': statY_rmse,
        'r_3sig': r_3sig,
        'r_actual': r_actual
    })

    # Print results
    print(f'CCR #{ccNum}')
    print(f'{"Parameter":<20}{"Measured":<20}{"{} Sigma".format(sigma):<15}{"CCR":<15}')
    print(f'{"---------":<20}{"----------":<20}{"-----------":<15}{"----------":<15}')
    print(f'{"Chord Length (m)":<20}{cc_Struct["chord"]:.2f}{cc_Struct["chordNumPts"]}{"":<7}{cc_Struct["chordPred"]:.2f}')
    print(f'{"X Centroid (m)":<20}{cc_Struct["xCenterLeftRot"]:<20.2f} {cc_Struct["xCenterLeftRotPred"]:<15.2f}{ccXrot_truth.item():<15.2f}')
    print(f'{"Y Centroid (m)":<20}{cc_Struct["yCenterLeftRot"]:<20.2f} {cc_Struct["yCenterLeftRotPred"]:<15.2f}{ccYrot_truth.item():<15.2f}')
    print(f'{"Delta X (m)":<20}{cc_Struct['xCentroidDeltaLeft'].item():<20.2f} {cc_Struct['xCentroidDeltaLeftPred'].item():<15.2f}{"":<15}')
    print(f'{"Delta Y (m)":<20}{cc_Struct['yCentroidDeltaLeft'].item():<20.2f} {cc_Struct['yCentroidDeltaLeftPred'].item():<15.2f}{"":<15}')

    return cc_Struct, plot_data

def get_combos(cc_Struct_dict, centroidSides, ccrX_truthRot_dict, ccrY_truthRot_dict):
    # Initialize the combo arrays
    combo_array = []
    combo_arrayPred = []

    # Get the number of keys (or current_index) in cc_Struct_dict
    num_indices = len(cc_Struct_dict)

    # Generate all combinations of centroidSides based on the number of indices
    centroid_combinations = list(itertools.product(centroidSides, repeat=num_indices))

    # Loop over each combination
    for combo in centroid_combinations:
        combo_row = [None] * 46
        combo_row_pred = [None] * 46

        # Access the correct structure for each index
        indices = list(cc_Struct_dict.keys())
        x_offsets, y_offsets, x_offsets_pred, y_offsets_pred = [], [], [], []

        for i, side in enumerate(combo):
            current_index = indices[i]
            cc_Struct = cc_Struct_dict[current_index]

            # Construct the combination entry
            combo_row[i] = f"{current_index}*{side}"
            combo_row_pred[i] = f"{current_index}*{side}"

            # Retrieve X and Y centroid deltas and predictions
            x_offsets.append(np.real(cc_Struct.get(f'xCentroidDelta{side}', 0)))
            y_offsets.append(np.real(cc_Struct.get(f'yCentroidDelta{side}', 0)))
            x_offsets_pred.append(np.real(cc_Struct.get(f'xCentroidDelta{side}Pred', 0)))
            y_offsets_pred.append(np.real(cc_Struct.get(f'yCentroidDelta{side}Pred', 0)))

        # Calculate mean offsets
        combo_row[18] = np.mean(x_offsets)
        combo_row[19] = np.mean(y_offsets)
        combo_row_pred[18] = np.mean(x_offsets_pred)
        combo_row_pred[19] = np.mean(y_offsets_pred)

        # Apply mean offsets in both X and Y
        for i, side in enumerate(combo):
            current_index = indices[i]
            cc_Struct = cc_Struct_dict[current_index]
            combo_row[20 + i] = np.real(cc_Struct.get(f'xCenter{side}Rot', 0)) + combo_row[18]
            combo_row[26 + i] = np.real(cc_Struct.get(f'yCenter{side}Rot', 0)) + combo_row[19]

            combo_row_pred[20 + i] = np.real(cc_Struct.get(f'xCenter{side}RotPred', 0)) + combo_row_pred[18]
            combo_row_pred[26 + i] = np.real(cc_Struct.get(f'yCenter{side}RotPred', 0)) + combo_row_pred[19]

        # Compute all residuals
        x_residuals = [combo_row[20 + i] - ccrX_truthRot_dict[indices[i]][0] for i in range(num_indices)]
        y_residuals = [combo_row[26 + i] - ccrY_truthRot_dict[indices[i]][0] for i in range(num_indices)]

        x_residuals_pred = [combo_row_pred[20 + i] - ccrX_truthRot_dict[indices[i]][0] for i in range(num_indices)]
        y_residuals_pred = [combo_row_pred[26 + i] - ccrY_truthRot_dict[indices[i]][0] for i in range(num_indices)]

        # Calculate RMSE for all residuals
        combo_row[44] = np.sqrt(np.mean(np.square(x_residuals)))  # RMSE X
        combo_row[45] = np.sqrt(np.mean(np.square(y_residuals)))  # RMSE Y
        combo_row_pred[44] = np.sqrt(np.mean(np.square(x_residuals_pred)))  # RMSE X
        combo_row_pred[45] = np.sqrt(np.mean(np.square(y_residuals_pred)))  # RMSE Y

        # Append to the output arrays
        combo_array.append(combo_row)
        combo_arrayPred.append(combo_row_pred)

    return combo_array, combo_arrayPred    

def get_footprint(ccr_data_index, gt_data, utm_correction, ccr_truth_data, footprint_range, output_dir):
    footprint_diameters_str = footprint_range
    start, step, end = map(float, footprint_diameters_str.split(':'))
    footprint_diameters = list(np.arange(start, end + step, step))  # Using NumPy for range generation
    results = []

    # Loop over footprint diameters
    for i, diameter in enumerate(footprint_diameters):
        
        r_nominal = diameter / 2
        
        print("\n")
        print(f"Diameter Used = {r_nominal * 2:.1f} m")
        print('----------------------')
        print("\n")
        
        # Assuming plot_ccr_footprints_ant is a defined function that returns multiple outputs
        results_temp, combo_arrayPred, cc_Struct_dict, direction, plot_data_list = get_offset(ccr_data_index, gt_data, r_nominal, ccr_truth_data, utm_correction)
        # Append the temporary results for this diameter to the results list
        results.append(results_temp)    
    
    # Get output results
    results = np.array(results)
    # Find the index of the minimum value in column index 5
    min_row_index = np.argmin(results[:, 5])  # Using index 5 for column 6 (0-based indexing)
    # Retrieve the nominal diameter
    nominal_diameter = results[min_row_index, 0]  # Column 1 (0-based index)

    # Print results
    print("\n\nRESULTS")
    print("-------")
    print(f"Nominal Footprint Diameter = {nominal_diameter:.2f} m\n")
    print(f"Statistical Easting Offset = {results[min_row_index, 1]:.2f} m")
    print(f"Statistical Northing Offset = {results[min_row_index, 2]:.2f} m\n")
    print(f"Statistical Cross-Track Offset = {results[min_row_index, 3]:.2f} m ({results[min_row_index, 5]:.2f} m RMSE)")
    print(f"Statistical Along-Track Offset = {results[min_row_index, 4]:.2f} m ({results[min_row_index, 6]:.2f} m RMSE)\n")

    #Initializing Values
    e_shift_pred = results[min_row_index, 1]  # Column 1 for Easting shift
    n_shift_pred = results[min_row_index, 2]  # Column 2 for Northing shift
    
    ccNumRegions = find_ccr_regions(ccr_data_index)
    first_valid_region = ccNumRegions['region_axes'].dropna().iloc[0]  # Get the first valid region
    x_limits = first_valid_region['x']
    y_limits = first_valid_region['y']

    # Plotting
    fig8 = plt.figure(figsize=(9, 6))
    plt.grid(True)
    plt.box(True)
    plt.axis('equal')
    plt.xlabel(ccr_truth_data['xlabelStr'].iloc[0])
    plt.ylabel(ccr_truth_data['ylabelStr'].iloc[0])

    # Create empty lists for handles and labels
    handles = []
    labels = []

    # Plot all CCR Arrays
    sc = plt.scatter(ccr_truth_data['ccrX'], ccr_truth_data['ccrY'], marker='x', color='gray', s=40)
    #Plot all CCR arrays text
    for current_index in range(len(ccr_truth_data['ccrNames'])):
        plt.text((ccr_truth_data['ccrX'].iloc[current_index]), (ccr_truth_data['ccrY'].iloc[current_index]-3), 
                ccr_truth_data['ccrNames'][current_index], color='gray', ha='center', va='center')
    handles.append(sc)
    labels.append('Other CCRs')

    # Plot GT2R and RGT data in Easting/Northing Plane
    gt_data_plot = plt.plot(gt_data['gt_x'], gt_data['gt_y'], 'k.', markersize=2)
    handles.append(gt_data_plot[0])
    labels.append('Ground Track Orginal')

    # Plot GT2R and RGT data with East/North Shift Predicted
    pred_gt_data_plot = plt.plot((gt_data['gt_x'] + e_shift_pred), (gt_data['gt_y'] + n_shift_pred), 'g.', markersize=2)
    handles.append(pred_gt_data_plot[0])
    labels.append('Ground Track Predicted')

    # Plot CCR Footprint (actual)
    for current_index, cc_struct in cc_Struct_dict.items():
        footprint_plot = plt.scatter((cc_struct['x'] + e_shift_pred), (cc_struct['y'] + n_shift_pred), marker='^', color=[128/255, 0/255, 128/255])
        if current_index == 0:  # Only add the label once
            handles.append(footprint_plot)
            labels.append('Actual CCR Returns')
    
    # Plot Footprints over Measured Data
    for current_index, cc_struct in cc_Struct_dict.items():
        for j in range(len(cc_struct['xRotPred'])):
            # Calculate the coordinates for the circle
            atl03CircleXrot, atl03CircleYrot = makeCircle((cc_struct['xPred'][j] + e_shift_pred),
                (cc_struct['yPred'][j] + n_shift_pred), cc_struct['r_3sig'])
            # Plot the circle
            plt.plot(atl03CircleXrot, atl03CircleYrot, '--', color=[0, 0.7, 0], linewidth=1)

    # Plot CCR footprint (predicted)
    for current_index, cc_struct in cc_Struct_dict.items():
        plt.scatter((cc_struct['xPred'] + e_shift_pred), (cc_struct['yPred'] + n_shift_pred),
        color='blue', s=64,  # Adjust markersize as needed
        facecolor='none', edgecolor='blue', linewidth=1)

        # Access the relevant x and y circle coordinates based on direction
        x_circle = (cc_struct[f'xCircle{direction[current_index]}Pred'] + e_shift_pred)
        y_circle = (cc_struct[f'yCircle{direction[current_index]}Pred'] + n_shift_pred)
        
        # Plot the circles as solid lines
        plt.plot(x_circle, y_circle, '-', color='b', linewidth=1)
    dummy_handle = plt.scatter([], [], color='blue', s=64, facecolor='none', edgecolor='blue', linewidth=1)  # Invisible point
    handles.append(dummy_handle)
    labels.append('2σ CCR Returns')

    # Plot CCR Hits
    for current_index, row in ccr_data_index.iterrows():
        closest_ccr_x = row['closest_ccr_x']  # Extract x for the current row
        closest_ccr_y = row['closest_ccr_y']  # Extract y for the current row
        hits = plt.plot(closest_ccr_x, closest_ccr_y, 'rx', markersize=10, markeredgewidth=3, markerfacecolor='r')
        if current_index == 0:
            handles.append(hits[0])
            labels.append('CCR Hit')

    # Plot Predicted CCR Location
    for current_index, cc_struct in cc_Struct_dict.items():
        x_center = (cc_struct[f'xCenter{direction[current_index]}Pred'] + e_shift_pred)
        y_center = (cc_struct[f'yCenter{direction[current_index]}Pred'] + n_shift_pred)
        
        # Plot the centers with specified styles
        pred_footprint_plot = plt.scatter(x_center, y_center, marker='s', color='b', s=100, 
                                       facecolor='none', edgecolor='b', linewidth=2, zorder=6)
        if current_index == 0:  # Only add the label once
            handles.append(pred_footprint_plot)
            labels.append('Predicted CCR Locations')

    # Plot CCR Hit Text
    for current_index, row in ccr_data_index.iterrows():
        closest_ccr_index = row['closest_ccr_index']
        plt.text((ccr_truth_data['ccrX'].iloc[closest_ccr_index]), (ccr_truth_data['ccrY'].iloc[closest_ccr_index]-3), 
                ccr_truth_data['ccrNames'][closest_ccr_index], color='red', fontweight='bold', ha='center', va='center')

    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.73)
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.savefig(os.path.join(output_dir, f'footprint_final.png'))
    plt.close(fig8)

    return results, plot_data_list

#Main function to execute everything
def full_output(h5_file_path, gt_num, region_name, footprint_range, imported_h5=None, run_select_data=True):
    # Extract the date from the H5 file name
    base_name = os.path.basename(h5_file_path)  # Get the file name from the full path
    date_str = base_name.split('_')[2][:8]  # Extracting date portion for naming

    # Create a unique identifier for the output directory using the current timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    output_dir = os.path.join("output", f"results_{date_str}_{timestamp}")  # Unique output directory

    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Get lidar output data
    lidar_mat_file = 'path_to_your_file/ATL03_data.mat'

    if run_select_data:
        # Full data processing
        gt_data_corrected, gt_data, utm_correction = read_and_transform_data(h5_file_path, gt_num, lidar_mat_file)
        ccr_truth_data = load_ccr_truth_data(region_name)

        # Plot the data using the plot_data function
        fig, ax1, scatter_plot = plot_selection_data(gt_data_corrected, gt_data, ccr_truth_data, region_name)

        # Handle user selections
        ccr_data_index = handle_selections(fig, ax1, scatter_plot, gt_data_corrected, ccr_truth_data)

        # Pass output_dir to get_footprint
        results, plot_data_list = get_footprint(ccr_data_index, gt_data, utm_correction, ccr_truth_data, footprint_range, output_dir)

        plot_figures(ccr_data_index, ccr_truth_data, gt_data_corrected, utm_correction, results, output_dir, plot_data_list)

        hdf5_filename = os.path.join(output_dir, f'ccrData_{date_str}.h5')
        export_to_hdf5(ccr_data_index, hdf5_filename)

    else:
        # Only import indices
        ccr_data_index = import_from_hdf5(imported_h5)  # Example file path
        
        # Re-run necessary data processing steps
        gt_data_corrected, gt_data, utm_correction = read_and_transform_data(h5_file_path, gt_num, lidar_mat_file)
        ccr_truth_data = load_ccr_truth_data(region_name)

        # Pass output_dir to get_footprint
        results, plot_data_list = get_footprint(ccr_data_index, gt_data, utm_correction, ccr_truth_data, footprint_range, output_dir)

        plot_figures(ccr_data_index, ccr_truth_data, gt_data_corrected, utm_correction, results, output_dir, plot_data_list)

        hdf5_filename = os.path.join(output_dir, f'ccrData_{date_str}.h5')
        export_to_hdf5(ccr_data_index, hdf5_filename)

    return ccr_data_index  # Return the index for further use if needed

    pass