from pathlib import Path

from flask import Flask, render_template

app = Flask(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

sample_images = [
    f"/static/{path.name}"
    for path in sorted(STATIC_DIR.glob("joined_idx_*.jpg"))[:3]
]

@app.route("/")
def index():
    return render_template('index.html', images=sample_images)

if __name__ == '__main__':
    app.run(debug=True)
