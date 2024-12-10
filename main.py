import customtkinter as ctk
from screens.EqualizerScreen import EqualizerScreen
from PIL import Image, ImageTk
import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

ctk.set_appearance_mode("Dark")
theme_path = resource_path("assets/dark-blue.json")
ctk.set_default_color_theme(theme_path)

app = ctk.CTk()
app.title("Giọng")
app.geometry("1100x500")
app.resizable(False, False)

try:
    if os.name == 'nt':
        icon_path = resource_path("assets/icon.ico")
        if os.path.exists(icon_path):
            app.iconbitmap(icon_path)
        else:
            print(f"Icon not found at: {icon_path}")
    else:
        icon_path = resource_path("assets/logo.png")
        if os.path.exists(icon_path):
            icon = ImageTk.PhotoImage(file=icon_path)
            app.iconphoto(False, icon)
        else:
            print(f"Logo not found at: {icon_path}")
except Exception as e:
    print(f"Error setting icon: {e}")

left_panel = ctk.CTkFrame(app, width=200)
left_panel.pack(side="left", fill="y")

logo_image = ctk.CTkImage(
    light_image=Image.open(resource_path("assets/logo.png")),
    dark_image=Image.open(resource_path("assets/logo.png")),
    size=(150, 150)
)
logo_label = ctk.CTkLabel(left_panel, image=logo_image, text="")
logo_label.pack(pady=20, padx=20) 

button_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
button_frame.pack(expand=True)

right_panel = ctk.CTkFrame(app)
right_panel.pack(side="right", expand=True, fill="both")

def switch_screen(screen):
    for widget in right_panel.winfo_children():
        widget.destroy()
    screen(right_panel)

switch_screen(EqualizerScreen)
app.mainloop()
