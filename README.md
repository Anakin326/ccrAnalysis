# ccrAnalysis

## Overview
ccrAnalysis is a Python package intended for users to analyze ATL03 geolocation offsets using the Corner Cube
Retroreflector (CCR) method. This repository provides both a command-line interface (CLI) and a graphical user interface (GUI) for interacting with the application. The application requires the use of Lidar return data files, typically in the .h5 format. The primary output is an output folder showing all of the results of the CCRanalysis, namely, the geolocation offsets and estimated ICESat-2 footprint diameter (along with RMSE values).

The repository includes all necessary scripts for data processing, visualization, and exporting results, except for the Lidar return data files.

## Repository Structure
The structure of the repository is as follows:

- **README.md**: Provides an overview of the repository, setup instructions, and usage details.
- **requirements.txt**: A file listing the required Python dependencies.
- **docs/**: Contains the API documentation generated with Sphinx.
- **src/**: The primary source code directory containing all the core logic for the project.
  - **main.py**: The main entry point for running the application.
  - **get_footprint.py**: Script for processing selected returns and calculating footprints.
  - **utils/**: Contains utility functions for data processing, visualization, and I/O operations.
    - **data_processing.py**: Module for cleaning and transforming Lidar data.
    - **plotting.py**: Module for creating visualizations of the data.
    - **file_io.py**: Manages input/output operations, including reading and writing `.h5` files.
- **supportFiles/**: Contains truth CCR data such as names, heights, etc.
- **ui/**: Contains scripts for the user interfaces.
  - **gui_app.py**: Graphical User Interface (GUI) script.
  - **cli_app.py**: Command Line Interface (CLI) script.

## Installation and Setup

### Clone the Repository

Start by cloning the repository to your local machine:
```
git clone https://github.com/Anakin326/ccrAnalysis.git
```
### Navigate to the repository directory:
```
cd ccrAnalysis
```
### Install Dependencies
Ensure you have Python 3.12 or higher installed on your system. It's recommended to set up a virtual environment for the project (optional but preferred). Then, install the required dependencies:
```
pip install -r requirements.txt
```
### Lidar Return Files
Ensure you have access to the Lidar return data files (typically .h5 format). These files are not included in the repository, so you'll need to provide them separately.

## Running the Application
The application provides both a GUI and CLI interface. You can choose between them when running the main.py script.

### Run with GUI
To run the application with the graphical user interface (GUI), use the following command:
```
python main.py gui
```
This will launch the GUI, where you can interact with the application through graphical prompts.

### Run with CLI
To run the application with the command-line interface (CLI), use the following command:
```
python main.py cli
```
If no mode (gui or cli) is specified, the system will prompt you to choose between the two interfaces.

## Usage
### GUI Flow
1. **Select .h5 File**: The GUI will prompt you to select the path to your .h5 Lidar return file. Click "Browse" to select the file from your local machine.

2. **Choose Ground Truth Number**: After selecting the file, you will be prompted to choose the ground truth number.

3. **Select Region and Footprint Range**: Next, you will select the region name and footprint range. The default options are:
   - Region: WSMR (White Sands Missile Range)
   - Footprint range: 5:0.1:20 (formatted as start:step:end)

4. **Select Data or Import Indices**:
   - You can either:- *Select* new data, or
   - *Import* previously saved indices if further validation is needed.
      
6. **Select CCR and Ground Photon Returns**: A new figure will appear for you to select the CCR return (higher altitude) and ground photon returns. The system will display the selections and their mean altitudes, and highlight the closest CCR return to your selected returns.

7. **Close the Selection Window**: After you have selected the data, close the selection figure window to begin the footprint calculation.

8. **Completion**: Once the footprint calculations are completed, the results will be displayed in the command line. The output will also include the location where your dataframe has been exported.

### CLI Flow
1. **Provide File Path**: The CLI will first prompt you to type the path to the .h5 file.

2. **Ground Truth Number**: After selecting the file, you will be asked to type the ground truth number.

3. **Region and Footprint Range**: You will be prompted to type the region name and footprint range.

4. **Import Indices**: When prompted with "Do you want to import indices?", you can respond with:
   - *Yes* to import previously saved indices, or
   - *No* to continue with the selection process.
5. **Select Data**: Similar to the GUI, you will select the CCR and ground photon returns. The CLI will display the selections and their mean altitudes for reference.

6. **Footprint Calculation and Completion**: After the calculations are completed, the results will be displayed in the terminal, and you will receive a success prompt along with the location of the exported dataframe.

## Output
Upon successful completion, an index of the selected CCR data, along with any generated figures, will be exported to a designated output directory. The output folder will be named using the following format:

```
results_<date_of_collection>_<date_of_processing>_<time_of_processing>
```

This naming convention ensures that multiple processes can be run on the same .h5 file without overwriting previous outputs.

## Troubleshooting
Missing Dependencies: If you encounter issues related to missing dependencies, ensure that all required packages are installed by running pip install -r requirements.txt again.

File Not Found: If you cannot find the .h5 Lidar return files, ensure that you have access to these files as they are required for the application to function.
