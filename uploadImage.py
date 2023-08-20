import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

UPLOAD_DIR = "images"
next_filename = 1003


def open_image():
    global next_filename

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))

        # Create the 'images' directory if it doesn't exist
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Save the image with the next filename
        filename = f"{next_filename}.png"
        img.save(os.path.join(UPLOAD_DIR, filename))

        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        # Show a success message
        messagebox.showinfo("Success", "Image uploaded successfully!")

        # Increment the filename for the next image
        next_filename += 1


# Create the main window
root = tk.Tk()
root.title("Upload Victim Image")

# Set the window size (width x height)
root.geometry("600x400")

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image, width=20, height=2)
open_button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Start the GUI event loop
root.mainloop()
