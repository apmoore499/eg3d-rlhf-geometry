from PIL import Image, ImageDraw
import os
import random

# Create a folder for dummy images
os.makedirs("dummy_images", exist_ok=True)

# Generate six dummy images with random colors
for i in range(1, 7):
    # Create a new image
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Generate random RGB color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Add a simple text in contrasting color to each image
    text = f"Image {i}"
    text_width, text_height = draw.textsize(text)
    text_color = (255 - color[0], 255 - color[1], 255 - color[2])
    draw.text(((200 - text_width) // 2, (200 - text_height) // 2), text, fill=text_color)

    # Save the image
    image.save(f"dummy_images/image{i}.jpg")

print("Dummy images with random colors created in the 'dummy_images' folder.")

