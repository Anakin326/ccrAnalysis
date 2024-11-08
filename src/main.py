import sys
from ui.gui_app import run_gui
from ui.cli_app import run_cli

def main():
    """
    Main function to run the application in either GUI or CLI mode.
    It checks the command-line arguments or prompts the user to choose 
    between the two modes (CLI or GUI).

    If the user provides an argument, it selects the mode based on that. 
    If no argument is provided, it prompts the user for input.
    
    The function then calls the appropriate function (`run_gui()` or `run_cli()`)
    depending on the user's choice.
    """
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input("Choose mode (CLI/GUI): ").strip().lower()

    if mode == "gui":
        run_gui()  # Call GUI mode
    elif mode == "cli":
        run_cli() # Call CLI mode
    else:
        print("Invalid mode. Use 'CLI' or 'GUI'.")

if __name__ == "__main__":
    main()