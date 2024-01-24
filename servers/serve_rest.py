import os

from flask import Flask, abort, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")
@app.route("/<id>", methods=["GET"])
def download_file(id):
    directory = os.path.join(data_path, id)
    filename = f"{id}.png"
    if os.path.isfile(os.path.join(directory, filename)):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        abort(404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
