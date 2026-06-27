from PIL import Image, ImageDraw
import os

# Create a folder for dummy images
os.makedirs("dummy_images", exist_ok=True)

# Generate six dummy images
for i in range(1, 7):
    # Create a new image
    image = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Add a simple text to each image
    text = f"Image {i}"
    text_width, text_height = draw.textsize(text)
    draw.text(((200 - text_width) // 2, (200 - text_height) // 2), text, fill=(0, 0, 0))

    # Save the image
    image.save(f"dummy_images/image{i}.jpg")

print("Dummy images created in the 'dummy_images' folder.")

