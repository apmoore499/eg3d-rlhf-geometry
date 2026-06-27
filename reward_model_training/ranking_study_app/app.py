from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List to store the order of clicked images
clicked_order = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_click', methods=['POST'])
def record_click():
    data = request.get_json()
    clicked_image = data['clicked_image']
    clicked_order.append(clicked_image)

    print(clicked_image)

    return jsonify({'success': True})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
