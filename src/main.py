import sys
from ui.gui_app import run_gui
from ui.cli_app import run_cli

def main():
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