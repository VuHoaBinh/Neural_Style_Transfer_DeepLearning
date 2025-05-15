from flask import Flask, request, send_file, jsonify
import os
import uuid
from model import load_model
from style_transfer import stylize_video

app = Flask(__name__)
os.makedirs("tmp", exist_ok=True)

model, device = load_model()


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    input_path = f"tmp/{uuid.uuid4()}.mp4"
    output_path = f"tmp/{uuid.uuid4()}_styled.mp4"
    file.save(input_path)

    try:
        stylize_video(input_path, output_path, model, device)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return send_file(output_path, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
