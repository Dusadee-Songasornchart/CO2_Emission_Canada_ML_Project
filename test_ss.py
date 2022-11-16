import io
import os

import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import Column, HorizontalSeparator, In, VSeperator

from PIL import Image

import cloudscraper

url = "https://cdnb.artstation.com/p/users/avatars/000/149/439/large/fe2b0699a4a2db62eb2814d44c81a0cf.jpg"
jpg_data = (
    cloudscraper.create_scraper(
        browser={"browser": "firefox", "platform": "windows", "mobile": False}
    )
    .get(url)
    .content
)

pil_image = Image.open(io.BytesIO(jpg_data))
png_bio = io.BytesIO()
pil_image.save(png_bio, format="PNG")
png_data = png_bio.getvalue()

imgViewer = [
    [sg.Image(data=png_data, key="-ArtistAvatarIMG-")],
    [sg.Button("Get Cover Image", key="Clicky")],
]