import os

from flask import Flask, abort, send_from_directory

app = Flask(__name__)


@app.route("/<id>", methods=["GET"])
def download_file(id):
    directory = f"data/{id}"
    filename = f"{id}.png"
    if os.path.isfile(os.path.join(directory, filename)):
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        abort(404)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
