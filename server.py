from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS, cross_origin
import os
import tempfile
import cv2
import face_recognition
import time
from openai import OpenAI
import gtts
import io

app = Flask(__name__, static_folder='my-portfolio/build', static_url_path='')
CORS(app)

# Define directories
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
IMAGE_DIRECTORY = os.path.join(os.path.dirname(__file__), 'images')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)

# Initialize known faces
known_face_encodings = []
known_face_names = []
client = OpenAI(api_key=OPEN_AI_KEY)

# Global variable to control video capture
camera_active = False
cap = None
latest_audio_url = None
latest_tts_text = "Default text"

# Load known faces
def load_known_faces():
    for filename in os.listdir(IMAGE_DIRECTORY):
        if filename.endswith(".jpeg"):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(IMAGE_DIRECTORY, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(name)

load_known_faces()

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    print("upload attempt")

    file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        image = face_recognition.load_image_file(temp_file.name)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(file.filename.split('.')[0])

    os.remove(temp_file.name)

    return jsonify({"message": "Image uploaded and face registered."})

def recognize_faces(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        names.append(name)
    return names

def generate_frames():
    print("generating frames")
    global cap, camera_active, latest_tts_text
    if not camera_active:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
    last_response_time = 0
    response_text = "No faces recognized."
    frame_skip = 5
    frame_count = 0

    while camera_active:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read a frame from the camera.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        recognized_names = recognize_faces(frame)

        current_time = time.time()
        if current_time - last_response_time >= 10:
            if recognized_names:
                response_text = get_chatgpt_response(recognized_names)
            else:
                response_text = "I don't recognize you."

            latest_tts_text = response_text  # Update the global variable with the latest text
            last_response_time = current_time

        for (top, right, bottom, left), name in zip(face_recognition.face_locations(frame), recognized_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(frame, response_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_latest_text', methods=['GET'])
@cross_origin()
def get_latest_text():
    global latest_tts_text
    return jsonify({"text": latest_tts_text})

@app.route('/tts', methods=['POST'])
@cross_origin()
def tts():
    global latest_tts_text
    request_data = request.get_json()
    text = request_data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tts = gtts.gTTS(text, lang='en')
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)

    global latest_tts_text
    latest_tts_text = text

    return send_file(
        audio_stream,
        mimetype='audio/mpeg',
        as_attachment=False,
        download_name='output.mp3'
    )

@app.route('/get_latest_audio', methods=['GET'])
@cross_origin()
def get_latest_audio():
    if latest_audio_url:
        return jsonify({"audio_url": latest_audio_url})
    else:
        return jsonify({"audio_url": None})

@app.route('/video_feed')
@cross_origin()
def video_feed():
    print("mark video feed")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video', methods=['POST'])
@cross_origin()
def start_video():
    print("start feed")
    global camera_active
    camera_active = True
    return jsonify({"message": "Video feed started."})

@app.route('/stop_video', methods=['POST'])
@cross_origin()
def stop_video():
    global camera_active
    camera_active = False
    return jsonify({"message": "Video feed stopped."})

def get_chatgpt_response(recognized_names):
    names_string = ", ".join(recognized_names)
    prompt = f"I see the following people: {names_string}. What would a dog say to them?"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":
             "You are a dog with the new ability to speak. Speak in broken English as a dog would. When you see a person, address them by their name and give a rude or vulgar comment as a dog might do. Keep each comment brief, no longer than one sentence. An example may be 'Hi (name), Can I hump on your leg?' or 'Oh, (name), you smell like dog food today.' Come up with some of your own and use my examples from time to time. Only ask to sniff someones butt occasionally. You may also demand treats or a walk. Riley is another dog you live with and may make comments about her too."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

@app.route('/')
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)