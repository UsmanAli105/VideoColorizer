from flask import Flask, request, jsonify, send_file
import os
from flask_cors import CORS
from color import main_colorize

app = Flask(__name__)
CORS(app)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if not file.content_type.startswith('video/'):
        return jsonify({'error': 'Invalid file type. Only video files are allowed.'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.mp4')
        file.save(filepath)
        return jsonify({'message': f'File {file.filename} uploaded successfully!'}), 200
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    


@app.route('/process', methods=['POST'])
def process_video():
    try:
        main_colorize()
        return jsonify({'message': 'Video processed successfully!'}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': f'An error occurred during video processing: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download_video():
    try:
        output_path = 'output/output.mp4'
        if not os.path.exists(output_path):
            return jsonify({'message': 'File not found'}), 404
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
