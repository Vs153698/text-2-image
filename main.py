import tkinter as tk
from tkinter import scrolledtext, filedialog
from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

# Log in to Hugging Face
# openskyml/midjourney-mini
login(token="your_hugging_face_token")

# Load the Stable Diffusion model with mixed-precision
model_name = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe = pipe.to("mps")  # Use Apple Silicon (Metal Performance Shaders) for faster processing

def generate_image():
    text = input_text.get("1.0", tk.END).strip()
    if text:
        # Generate image
        with torch.no_grad():
            image = pipe(text).images[0]

        # Resize image for display
        display_image = image.resize((256, 256), Image.Resampling.LANCZOS)

        # Display image
        img = ImageTk.PhotoImage(display_image)
        output_image_label.config(image=img)
        output_image_label.image = img

        # Show the HD image in a fullscreen window
        show_hd_image(image)

def show_hd_image(image):
    # Create a new fullscreen window
    hd_window = tk.Toplevel(root)
    hd_window.attributes('-fullscreen', True)
    hd_window.title("HD Image Display")

    # Display the HD image
    hd_img = ImageTk.PhotoImage(image)
    hd_label = tk.Label(hd_window, image=hd_img)
    hd_label.image = hd_img  # Keep a reference to avoid garbage collection
    hd_label.pack()

    # Add a download button
    download_button = tk.Button(hd_window, text="Download Image", command=lambda: download_image(image))
    download_button.pack()

    # Add an exit fullscreen button
    exit_button = tk.Button(hd_window, text="Exit Fullscreen", command=hd_window.destroy)
    exit_button.pack()

def download_image(image):
    # Open file dialog to save the image
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        image.save(file_path)

# Set up the main application window
root = tk.Tk()
root.title("Text to Image Generator")
root.geometry("600x400")

# Input text box
input_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10)
input_text.pack(pady=10)

# Generate button
generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=10)

# Output image label
output_image_label = tk.Label(root)
output_image_label.pack(pady=10)

# Run the application
root.mainloop()
