# src/utils/plotting.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromCollection:
    """
    Class to handle lasso selection on scatter plots to select points.
    """
    def __init__(self, ax, collection, alpha_other=0.3):
        """
        Initialize the lasso selector for scatter plot selection.

        Args:
            ax (matplotlib.axes.Axes): The axis where the plot is drawn.
            collection (matplotlib.collections.PathCollection): The scatter plot collection.
            alpha_other (float): The transparency level for non-selected points (default 0.3).
        """
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)
        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

    def onselect(self, verts):
        """
        Handle the lasso selection event.

        Args:
            verts (list of tuple): Vertices of the selected region.
        """
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        """
        Disconnect the lasso selector and restore original point colors.
        """
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
    pass

def makeCircle(x, y, r):
    """
    Generate the points for a circle given its center and radius.

    Args:
        x (float): x-coordinate of the center.
        y (float): y-coordinate of the center.
        r (float): Radius of the circle.

    Returns:
        tuple: Two numpy arrays representing the x and y coordinates of the circle.
    """
    theta = np.linspace(0, 2 * np.pi, 100)  # Same as MATLAB: th = 0:pi/50:2*pi
    xPts = r * np.cos(theta) + x
    yPts = r * np.sin(theta) + y
    return xPts, yPts

def find_ccr_regions(region_name, ccr_data_index):
    """
    Assign each entry in the CCR data to a specific region based on the ccNum.

    Args:
        ccr_data_index (pandas.DataFrame): DataFrame containing the CCR data with 'ccNum' column.

    Returns:
        pd.DataFrame: DataFrame with each CCR number, assigned region, and region's x/y limits.
    """
    # Define the region_axes dictionary
    region_axes = { 
        'north': {'x': (367240, 367340), 'y': (3651070, 3651170)},
        'east': {'x': (367260, 367360), 'y': (3650630, 3650700)},
        'south': {'x': (367230, 367330), 'y': (3650170, 3650250)},
        'west': {'x': (367180, 367260), 'y': (3650630, 3650710)},
    }

    ccNumRegionAll = []
    for index, row in ccr_data_index.iterrows():
        ccNum = row['ccNum']  # Access the ccNum from the current row
        try:
            # Check if regionName is 'wsmr' (case-insensitive)
            if region_name.lower() == 'wsmr':
                # Assign regions based on the numeric value of ccNum
                if 1 <= ccNum <= 12:
                    region = 'north'
                elif 13 <= ccNum <= 24:
                    region = 'east'
                elif 25 <= ccNum <= 36:
                    region = 'south'
                elif 37 <= ccNum <= 48:
                    region = 'west'
                else:
                    region = 'unknown'  # Fallback for numbers outside the expected range
            
                ccNumRegionAll.append((ccNum, region, region_axes.get(region, None)))  # Append ccNum, region, and axes
            else:
                # For other region names, assign AR + the first character of ccNum
                ccNumRegionAll.append(('AR' + str(ccNum)))  # Assuming ccNum is a string

        except ValueError as e:
            print(f"Error: {e}")
            ccNumRegionAll.append((ccNum, 'unknown', None))  # Append unknown region

    # Convert to DataFrame for better visualization if needed
    ccNumRegions = pd.DataFrame(ccNumRegionAll, columns=['ccNum', 'region', 'region_axes'])
    
    return ccNumRegions

# Plot the data
def plot_selection_data(gt_data, ccr_truth_data, region_name):
    """
    Plot various data related to ground truth, CCR data, and selection regions.

    Args:
        gt_data (pandas.DataFrame): Ground truth data with raw coordinates.
        ccr_truth_data (pandas.DataFrame): Data of CCR truth points.
        region_name (str): The region name (e.g., 'WSMR') for plotting limits and labels.

    Returns:
        tuple: Matplotlib figure, axis, and scatter plot object.
    """

    minY, maxY = ccr_truth_data['ccrY'].min() - 20, ccr_truth_data['ccrY'].max() + 20
    minAlt, maxAlt = 1160, 1175
    
    fig = plt.figure(figsize=(10, 9))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    scatter_plot = ax1.scatter(gt_data['gt_y'], gt_data['gt_z'], s=1)
    ax1.set_xlabel(ccr_truth_data['ylabelStr'].iloc[0])  # Accessing the first element
    ax1.set_ylabel('Altitude (m)')
    ax1.grid(True)
    if region_name.lower() == 'wsmr':
        plt.axis([minY, maxY, minAlt, maxAlt])
    ax1.set_title('Sideview Plot of Photons')

    ax2 = plt.subplot2grid((2, 3), (1, 1))
    ax2.scatter(ccr_truth_data['ccrX'], ccr_truth_data['ccrY'], color='red', edgecolor='red')
    ax2.scatter(gt_data['gt_x'], gt_data['gt_y'], color='lime', s=1)
    for i, name in enumerate(ccr_truth_data['ccrNames']):
        ccrName = name[3:] if region_name.lower() == 'wsmr' else name
        ax2.text(ccr_truth_data['ccrX'].iloc[i], ccr_truth_data['ccrY'].iloc[i], ccrName, fontsize=5.5, ha='left', clip_on='True')
    ax2.set_xlabel(ccr_truth_data['xlabelStr'].iloc[0])
    ax2.set_ylabel(ccr_truth_data['ylabelStr'].iloc[0])
    if region_name.lower() == 'wsmr':
        ax2.set_ylim(minY, maxY)
    ax2.grid(True)
    ax2.set_title('Top-Down View', fontsize=10)

    def sync_xlimits(event):
        ax2.set_ylim(ax1.get_xlim())
        fig.canvas.draw_idle()

    ax1.callbacks.connect('xlim_changed', sync_xlimits)
    
    return fig, ax1, scatter_plot

def plot_figures(ccr_data_index, ccr_truth_data, gt_data, results, output_dir, plot_data_list, region_name):
    """
    Plots multiple figures to visualize and analyze the data, including ground truth, CCR data, and error analysis.

    Args:
        ccr_data_index (pandas.DataFrame): DataFrame containing CCR data including ground truth, ccrNum, and related fields.
        ccr_truth_data (pandas.DataFrame): DataFrame containing CCR truth data such as coordinates and labels.
        gt_data (pandas.DataFrame): DataFrame containing ground track data (e.g., UTM coordinates and altitude).
        results (numpy.ndarray): Results array with estimated footprint diameters and associated error metrics.
        output_dir (str): Directory to save the generated plots.
        plot_data_list (dict): Dictionary containing plot data for individual CCR returns and other information for histogram plotting.

    """
    # Apply UTM correction to ground track data
    measData_x = gt_data['gt_x']
    measData_y = gt_data['gt_y']
    measData_z = gt_data['gt_z']

    # Prepare for plotting all CCR and ground data in fig3 and fig4
    ccArray = np.empty((0, 2))

    # Find the regions for each CCR and extract plot limits from the first valid region
    ccNumRegions = find_ccr_regions(region_name, ccr_data_index)
    first_valid_region = ccNumRegions['region_axes'].dropna().iloc[0]  # Get the first valid region
    x_limits = first_valid_region['x']
    y_limits = first_valid_region['y']

    # Iterate over all saved CCRs in the dictionary and plot individual graphs for each ccNum
    for index, row in ccr_data_index.iterrows():
        ccNum = row['ccNum']
        ccrData = row['ccrData']
        groundData = row['groundData']

        # --- Plot 2: Plot Height Data for each CCR ---
        fig2 = plt.figure()
        plt.plot(gt_data['gt_y'], gt_data['gt_z'], 'y.', label='All Points')
        plt.plot(ccrData[:, 0], ccrData[:, 1], 'r.', label=f'CCR Data (CCR {ccNum})')
        plt.plot(groundData[:, 0], groundData[:, 1], 'b.', label='Ground Data')
        plt.grid(True)
        plt.box(True)
        plt.xlabel(ccr_truth_data['xlabelStr'].iloc[0])
        plt.ylabel('Altitude (m)')
        plt.title(f'CCR {ccNum}', pad=20)
        plt.legend()
        plt.axis([min(groundData[:, 0]) - 2, max(groundData[:, 0]) + 2, min(groundData[:, 1]) - 2, max(ccrData[:, 1]) + 2])
        plt.savefig(os.path.join(output_dir, f'CCR_{ccNum}_altitude_plot.png'))
        plt.close(fig2)  # Close the figure to free memory
    
    for index, row in ccr_data_index.iterrows():
        ccrData = row['ccrData']
        groundData = row['groundData']
        closest_ccr_name = row['closest_ccr_name']
        closest_ccr_index = row['closest_ccr_index']  # Assuming the numeric index is stored in the dictionary

        # Apply UTM correction to CCR data
        corrected_ccrData = np.copy(ccrData)
        ccArray = np.vstack((ccArray, ccrData))

        # Find the indices where both y and z match in measData and ccArray
        inds = np.isin(measData_y, ccArray[:, 0]) & np.isin(measData_z, ccArray[:, 1])
        # Extract corresponding x and y values from measData where inds is True
        ccrX_ = measData_x[inds]
        ccrY_ = measData_y[inds]

        # Calculate plot limits
        maxCCRy = (max(ccr_truth_data['ccrY']) + 10)
        minCCRy = (min(ccr_truth_data['ccrY']) - 10)

    # --- Plot 3: Combined Track Plot for All CCRs ---
    fig3 = plt.figure()
    plt.scatter(measData_x, measData_y, color='lime', s=1, label='Ground Track')
    plt.scatter(ccrX_, ccrY_, color='blue', s=1, label='All CCR Returns')
    #Plot all CCRs and highlight the closest CCR for each entry in ccr_data_index
    for index, row in ccr_data_index.iterrows():
        ccrData = row['ccrData']
        closest_ccr_index = row['closest_ccr_index']  # Use the numeric index to highlight closest CCR

        # Plot all CCR points in gray
        plt.scatter(ccr_truth_data['ccrX'], ccr_truth_data['ccrY'], color='gray', s=1)

        # Highlight the closest CCR in red
        closest_ccr_x = ccr_truth_data['ccrX'].iloc[closest_ccr_index]
        closest_ccr_y = ccr_truth_data['ccrY'].iloc[closest_ccr_index]
        plt.scatter(closest_ccr_x, closest_ccr_y, color='red', s=5, label=f'Closest: {row["closest_ccr_name"]}')

    plt.axis('equal')
    plt.box(True)
    plt.grid(True)
    plt.xlabel(ccr_truth_data['xlabelStr'].iloc[0])
    plt.ylabel(ccr_truth_data['ylabelStr'].iloc[0])
    plt.ylim([minCCRy, maxCCRy])
    plt.title('Combined Track Plot for All CCRs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_track_plot.png'))
    plt.close(fig3)

    # --- Plot 4: Top-view Plot for All CCRs ---
    fig4 = plt.figure()
    plt.plot(measData_x, measData_y, 'o', color='lime', label='Ground Track')
    plt.plot(ccrX_, ccrY_, 'o', color='blue', label='CCR Returns')

    # Set to collect indices of selected CCRs
    selected_ccr_indices = ccr_data_index['closest_ccr_index'].tolist()

    # Plot all CCRs in gray and label those not in selected
    for j in range(len(ccr_truth_data)):
        plt.plot(ccr_truth_data['ccrX'].iloc[j], ccr_truth_data['ccrY'].iloc[j], 'o', color='gray')
        if j not in selected_ccr_indices:
            plt.text(ccr_truth_data['ccrX'].iloc[j] - 5, ccr_truth_data['ccrY'].iloc[j] - 3, ccr_truth_data['ccrNames'].iloc[j], color='gray')
            plt.text(ccr_truth_data['ccrX'].iloc[j] - 4, ccr_truth_data['ccrY'].iloc[j] - 6, f"{ccr_truth_data['ccrRelativeHeights'].iloc[j]:.1f} ft", color='gray')

    for index, row in ccr_data_index.iterrows():
        closest_ccr_index = row['closest_ccr_index']
        plt.plot(ccr_truth_data['ccrX'].iloc[closest_ccr_index], ccr_truth_data['ccrY'].iloc[closest_ccr_index], 'o', color='red')
        plt.text(ccr_truth_data['ccrX'].iloc[closest_ccr_index] - 5, ccr_truth_data['ccrY'].iloc[closest_ccr_index] - 3, row["closest_ccr_name"], color='red', fontweight='bold')
        plt.text(ccr_truth_data['ccrX'].iloc[closest_ccr_index] - 4, ccr_truth_data['ccrY'].iloc[closest_ccr_index] - 6, f"{row['closest_ccr_height']:.1f} ft", color='red', fontweight='bold')

    plt.axis('equal')
    plt.box(True)
    plt.grid(True)
    plt.xlabel(ccr_truth_data['xlabelStr'].iloc[0])
    plt.ylabel(ccr_truth_data['ylabelStr'].iloc[0])
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'top_view_plot.png'))
    plt.close(fig4)

    # --- Plot 5: Error Analysis (Cross-Track RMSE, Cross-Track Offset, Along-Track Offset) ---
    fig5, axs = plt.subplots(3, 1, figsize=(12, 9))

    min_row_index = np.argmin(results[:, 5])  # Find the index of the minimum value in column index 5
    nominal_diameter = results[min_row_index, 0]  # Finding nominal diameter

    # Plot for Cross-Track RMSE
    axs[0].plot(results[:, 0], results[:, 5], 'b-')
    ax_limits = axs[0].axis()
    axs[0].axvline(x=results[min_row_index, 0], color='k', linestyle=':')
    p1 = axs[0].plot(results[min_row_index, 0], results[min_row_index, 5], 'ko', markerfacecolor='r', markersize=8, linewidth=2)
    axs[0].set_ylabel('Cross-Track\nRMSE (m)')
    axs[0].set_title(f"Error Analysis\nEstimated Nominal Footprint Diameter = {nominal_diameter:.1f} m", fontsize=10)
    axs[0].legend(p1, ['Min Error'])
    axs[0].grid(True)
    axs[0].axis('tight')

    # Plot for Cross-Track Offset
    axs[1].plot(results[:, 0], results[:, 3], 'b-')
    ax_limits = axs[1].axis()
    axs[1].axvline(x=results[min_row_index, 0], color='k', linestyle=':')
    axs[1].plot(results[min_row_index, 0], results[min_row_index, 3], 'ko', markerfacecolor='r', markersize=8, linewidth=2)
    axs[1].set_ylabel('Cross-Track\nOffset (m)')
    axs[1].set_title(f"Cross-Track Offset = {results[min_row_index, 3]:.2f} m", fontsize=10)
    axs[1].grid(True)
    axs[1].axis('tight')

    # Plot for Along-Track Offset
    axs[2].plot(results[:, 0], results[:, 4], 'b-')
    ax_limits = axs[2].axis()
    axs[2].axvline(x=results[min_row_index, 0], color='k', linestyle=':')
    axs[2].plot(results[min_row_index, 0], results[min_row_index, 4], 'ko', markerfacecolor='r', markersize=8, linewidth=2)
    axs[2].set_xlabel('Footprint Diameter (m)')
    axs[2].set_ylabel('Along-Track\nOffset (m)')
    axs[2].set_title(f"Along-Track Offset = {results[min_row_index, 4]:.2f} m", fontsize=10)
    axs[2].grid(True)
    axs[2].axis('tight')

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis_plots.png'))
    plt.close(fig5)

    # --- Plot 6: Cross-Track RMSE (Single Plot) ---
    fig6 = plt.figure()
    plt.plot(results[:, 0], results[:, 5], 'b-', label='Cross-Track RMSE')  # 'b-' is the blue line
    # Highlight the minimum row
    p1 = plt.plot(results[min_row_index, 0], results[min_row_index, 5], 'ko', 
                   markerfacecolor='r', markersize=8, linewidth=2)  # 'ko' is the black circle
    # Set the labels and title
    plt.ylabel('Cross-Track\nRMSE (m)')
    plt.xlabel('Footprint Diameter (m)')
    plt.title('Error Analysis')  # Use f-string for dynamic title
    # Add the legend
    plt.legend([p1[0]], [f'Footprint Diameter = {nominal_diameter:.1f} m'])  # Using formatted string for diameter
    plt.grid(True)
    plt.box(True)
    plt.savefig(os.path.join(output_dir, 'cross_track_rmse_analysis.png'))
    plt.close(fig6)  # Close the figure to free memory

    # --- Plot 7: Histogram of CCR Photon Returns ---
    for current_index, plot_data in plot_data_list.items():
        ccNum = plot_data["ccNum"]
        fig7 = plt.figure()
        
        # Create histogram for CCR Photon Returns
        plt.hist(plot_data["ccYrot"], density=True, alpha=0.6, edgecolor='black', label='CCR Photon Returns')
        plt.plot(plot_data["ccYrot"], np.zeros_like(plot_data["ccYrot"]), 'ko')  # Black circles at y=0

        # Sort Gaussian points by the first column (distPts)
        gaussPtsUnordered = np.column_stack((plot_data["distPts"], plot_data["y2"]))
        gaussPtsOrdered = gaussPtsUnordered[np.argsort(gaussPtsUnordered[:, 0])]

        # Plot Gaussian fit
        plt.plot(gaussPtsOrdered[:, 0], gaussPtsOrdered[:, 1], '-', linewidth=3, label='Gaussian Fit')

        # Plot the mean and standard deviation markers
        plt.plot(plot_data["histMean"], 0, 'r*', label='Mean')
        for i in range(1, 4):
            plt.plot(plot_data["histMean"] + i * plot_data["histStd"], 0, 'b*')
            plt.plot(plot_data["histMean"] - i * plot_data["histStd"], 0, 'b*')
        
        # Plot predicted and actual min/max points
        plt.plot(plot_data["yMinPredictedRot"], 0, 'm*', label=f'{plot_data["sigma"]:.2f} σ ({plot_data["pctTotBoundPred"]:.2f}%)')
        plt.plot(plot_data["yMaxPredictedRot"], 0, 'm*')
        plt.plot(plot_data["yMinActualRot"], 0, 'g*', label=f'{plot_data["actualTotSigma"]:.2f} σ ({plot_data["pctTotBound"]:.2f}%)')
        plt.plot(plot_data["yMaxActualRot"], 0, 'g*')

        # Plot e2 min/max points with corresponding thresholds
        plt.plot(plot_data["e2_minX"], 0, 'ks', label='1/e^2 Range')
        plt.plot(plot_data["e2_maxX"], 0, 'ks')
        plt.plot(plot_data["e2_minX"], plot_data["e2_thresh"], 'ks')
        plt.plot(plot_data["e2_maxX"], plot_data["e2_thresh"], 'ks')

        # Dashed lines for e2 range
        plt.plot([plot_data["e2_minX"], plot_data["e2_minX"]], [0, plot_data["e2_thresh"]], 'k--')
        plt.plot([plot_data["e2_minX"], plot_data["e2_maxX"]], [plot_data["e2_thresh"], plot_data["e2_thresh"]], 'k--')
        plt.plot([plot_data["e2_maxX"], plot_data["e2_maxX"]], [plot_data["e2_thresh"], 0], 'k--')

        plt.grid(True)
        plt.box(True)
        plt.xlabel('Along-Track (m)')
        plt.ylabel('Probability Density')
        plt.title(f'CCR #{ccNum} Returns')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'CCR_{ccNum}_Hist.png'))
        plt.close(fig7)