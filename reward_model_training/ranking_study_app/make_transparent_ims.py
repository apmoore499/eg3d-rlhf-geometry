from PIL import Image
import os
from pathlib import Path

def make_images_transparent(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # try:
            # Open the image
            img = Image.open(file_path)

            # Convert to RGBA (if not already in RGBA mode)
            img = img.convert("RGBA")

            # Get data as a list of tuples
            data = list(img.getdata())

            # Make all pixels transparent
            transparent_data = [(r, g, b, 0) for r, g, b, a in data]

            # Put the new data back into the image
            img.putdata(transparent_data)

            file_path=file_path.replace('.jpg','.png')

            # Save the new image
            img.save(file_path)

            print(f"Image '{filename}' made transparent.")

            # except Exception as e:
            #     print(f"Error processing image '{filename}': {str(e)}")

if __name__ == "__main__":
    folder_path = Path(__file__).resolve().parent / "static" / "dummy_images_transparent"
    make_images_transparent(str(folder_path))
