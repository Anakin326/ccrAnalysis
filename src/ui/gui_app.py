# src/gui/gui_app.py
import tkinter as tk
from get_footprint import full_output
from tkinter import filedialog, messagebox

class DataProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing GUI")

        # Create and configure the main frame
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        # Define widgets and their layout
        tk.Label(frame, text="H5 File Path:").grid(row=0, column=0, sticky='e', padx=5)
        self.entry_h5_file = tk.Entry(frame, width=40)
        self.entry_h5_file.grid(row=0, column=1, padx=5)

        btn_browse = tk.Button(frame, text="Browse", command=self.browse_h5_file)
        btn_browse.grid(row=0, column=2, padx=5)

        tk.Label(frame, text="Ground Truth Number:").grid(row=1, column=0, sticky='e', padx=5)
        self.entry_gt_num = tk.Entry(frame)
        self.entry_gt_num.grid(row=1, column=1, padx=5)

        # Region Name Radio Buttons
        tk.Label(frame, text="Region Name:").grid(row=2, column=0, sticky='e', padx=5)
        self.selected_region = tk.StringVar(value="wsmr")  # Default value
        radio_frame = tk.Frame(frame)
        radio_frame.grid(row=2, column=1, columnspan=2)

        radio_wsmr = tk.Radiobutton(radio_frame, text="WSMR", variable=self.selected_region, value="wsmr")
        radio_wsmr.pack(side=tk.LEFT, padx=5)

        radio_antarctic = tk.Radiobutton(radio_frame, text="Antarctic", variable=self.selected_region, value="antarctic")
        radio_antarctic.pack(side=tk.LEFT, padx=5)

        tk.Label(frame, text="Footprint Range:").grid(row=3, column=0, sticky='e', padx=5)
        self.entry_footprint_range = tk.Entry(frame)
        self.entry_footprint_range.grid(row=3, column=1, padx=5)
        self.entry_footprint_range.insert(0, '5:0.1:20')  # Default value

        # Action buttons
        button_frame = tk.Frame(frame)
        button_frame.grid(row=4, columnspan=3, pady=20)
        btn_select_data = tk.Button(button_frame, text="Select Data", command=self.run_select_data)
        btn_select_data.pack(side=tk.LEFT, padx=10)

        btn_import_indices = tk.Button(button_frame, text="Import Indices", command=self.run_import_indices)
        btn_import_indices.pack(side=tk.LEFT, padx=10)

    # GUI helper functions within the class
    def browse_h5_file(self):
        file_path = filedialog.askopenfilename(title="Select HDF5 file", filetypes=[("HDF5 files", "*.h5")])
        if file_path:
            self.entry_h5_file.delete(0, tk.END)  # Clear the current entry
            self.entry_h5_file.insert(0, file_path)  # Insert the selected file path

    def run_select_data(self):
        h5_file_path = self.entry_h5_file.get()
        gt_num = self.entry_gt_num.get()
        region_name = self.selected_region.get()
        footprint_range = self.entry_footprint_range.get()
        
        try:
            full_output(h5_file_path, gt_num, region_name, footprint_range, run_select_data=True)
            messagebox.showinfo("Success", "Data selection completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select data: {str(e)}")

    def run_import_indices(self):
        gt_num = self.entry_gt_num.get()
        region_name = self.selected_region.get()
        footprint_range = self.entry_footprint_range.get()
        
        imported_h5 = filedialog.askopenfilename(title="Select HDF5 file", filetypes=[("HDF5 files", "*.h5")])
        
        if imported_h5:
            try:
                h5_file_path = self.entry_h5_file.get()
                full_output(h5_file_path, gt_num, region_name, footprint_range, imported_h5=imported_h5, run_select_data=False)
                messagebox.showinfo("Success", "Indices imported and processed successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process data: {str(e)}")
        else:
            messagebox.showwarning("Warning", "No file selected. Please select an HDF5 file.")
    pass

# Entry function to start the GUI
def run_gui():
    root = tk.Tk()
    app = DataProcessingGUI(root)
    root.mainloop()

# def main():
#     parser = argparse.ArgumentParser(description="Data Processing CLI")
    
#     parser.add_argument("h5_file_path", help="Path to the HDF5 file")
#     parser.add_argument("gt_num", type=int, help="Ground truth number")
#     parser.add_argument("region_name", choices=["wsmr", "antarctic"], help="Region name")
#     parser.add_argument("footprint_range", help="Footprint range, e.g., '5:0.1:20'")
#     parser.add_argument("--imported_h5", help="Path to an imported HDF5 file for indices", default=None)

#     args = parser.parse_args()
    
#     # Determine the mode based on the presence of --imported_h5
#     run_select_data = args.imported_h5 is None
    
#     try:
#         full_output(
#             h5_file_path=args.h5_file_path,
#             gt_num=args.gt_num,
#             region_name=args.region_name,
#             footprint_range=args.footprint_range,
#             imported_h5=args.imported_h5,
#             run_select_data=run_select_data
#         )
#         print("Processing completed successfully!")
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")


# if __name__ == "__main__":
#     main()