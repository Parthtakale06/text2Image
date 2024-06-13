import tkinter as tk
import customtkinter as ctk 
from PIL import Image, ImageTk
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = ctk.CTk()  # Use customtkinter's CTk for the main window
app.geometry("560x560")
app.title("Image Generator")
ctk.set_appearance_mode("dark")  # Set the appearance mode to dark

# Create an entry widget
prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

# Create a frame for the generated image
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# Define the model
modelid = "CompVis/stable-diffusion-v1-4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pipeline with appropriate torch_dtype
if device.type == 'cuda':
    pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
else:
    pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token)

pipe.to(device) 

# Define the generate function
def generate(): 
    prompt_text = prompt.get()
    
    if device.type == 'cuda':
        with autocast(device.type): 
            result = pipe(prompt_text, guidance_scale=8.5)
    else:
        result = pipe(prompt_text, guidance_scale=8.5)
    
    print(result)  # Print the result to inspect its structure

    # Check if 'images' key exists and use it
    if 'images' in result:
        image = result['images'][0]
    else:
        raise KeyError("The key 'images' is not found in the pipeline output")

    image.save('generatedimage.png')
    img = Image.open('generatedimage.png')
    img = ImageTk.PhotoImage(img)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to the image

# Create a button to generate an image
trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()
