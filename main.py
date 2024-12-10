import customtkinter as ctk
from screens.EqualizerScreen import EqualizerScreen
from PIL import Image, ImageTk
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Initialize the application
ctk.set_appearance_mode("Dark")
# Use resource_path to get the correct path to the theme file
theme_path = resource_path("assets/dark-blue.json")
ctk.set_default_color_theme(theme_path)

app = ctk.CTk()
app.title("Giọng")
app.geometry("1100x500")
app.resizable(False, False)  # Disable both horizontal and vertical resizing

# Set window icon
try:
    if os.name == 'nt':  # Windows
        icon_path = resource_path("assets/icon.ico")  # Icon in root directory
        if os.path.exists(icon_path):
            app.iconbitmap(icon_path)
        else:
            print(f"Icon not found at: {icon_path}")
    else:  # Linux/Mac
        icon_path = resource_path("assets/logo.png")
        if os.path.exists(icon_path):
            icon = ImageTk.PhotoImage(file=icon_path)
            app.iconphoto(False, icon)
        else:
            print(f"Logo not found at: {icon_path}")
except Exception as e:
    print(f"Error setting icon: {e}")

# Define the layout
# Left panel with buttons
left_panel = ctk.CTkFrame(app, width=200)
left_panel.pack(side="left", fill="y")

logo_image = ctk.CTkImage(
    light_image=Image.open(resource_path("assets/logo.png")),
    dark_image=Image.open(resource_path("assets/logo.png")),
    size=(150, 150)  # Adjust the size of the logo
)
logo_label = ctk.CTkLabel(left_panel, image=logo_image, text="")
logo_label.pack(pady=20, padx=20) 

# Center the buttons in the left panel
button_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
button_frame.pack(expand=True)

# Right panel to display dynamic content
right_panel = ctk.CTkFrame(app)
right_panel.pack(side="right", expand=True, fill="both")

# Function to switch screens
def switch_screen(screen):
    # Clear the right panel
    for widget in right_panel.winfo_children():
        widget.destroy()

    # Add the selected screen
    screen(right_panel)


switch_screen(EqualizerScreen)
app.mainloop()
