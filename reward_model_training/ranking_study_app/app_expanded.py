from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List to store the order of clicked images
clicked_order = []

# Total number of images (adjust this number based on your actual total)
total_images = 10

# Variable to store the current background image filename
current_bg_filename = "/path/to/initial_background.jpg"  # Replace with your initial background image path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/record_click", methods=["POST"])
def record_click():
    data = request.get_json()
    clicked_image = data["clicked_image"]
    clicked_order.append(clicked_image)
    console.log("total ims clicked")
    print(clicked_image)
    total_images = 5
    # Check if all images are clicked
    if len(clicked_order) == total_images:
        # Return a JSON response to inform JavaScript to change the background image
        response_data = {"success": True}

        return jsonify(response_data)

    return jsonify({"success": True})


@app.route("/change_background", methods=["POST"])
def change_background():
    global current_bg_filename  # Make sure to use the global variable

    data = request.get_json()
    new_bg_filename = data["new_bg_filename"]

    # Update the current background image filename
    current_bg_filename = new_bg_filename

    response_data = {"success": True}
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="127.0.0.0", port=5000, debug=True)
