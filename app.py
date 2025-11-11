from flask import Flask, flash, redirect, render_template, jsonify, request, session
from openai import OpenAI
from dotenv import load_dotenv
import os

# Reads the .env file which contains the key values
load_dotenv()
app = Flask(__name__)

# Setup the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup the session data
app.secret_key = os.getenv("SECRET_KEY")

# Loads the main webpage
@app.route("/")
def index():
    session.pop("previous_response_id", None)
    return render_template("index.html")

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    # Check if file is empty
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    # Save the user audio file
    save_path = os.path.join("static", "audio", "input.wav")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    # Internal testing (TO BE DELETED)
    print("Audio file saved:", save_path)

    # Transcribe file with Whisper
    with open(save_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_message = transcript.text
    
    # Internal testing (TO BE DELETED)
    print("Transcribed text:", user_message)

    # Allows for stateful conversations
    previous_id = session.get("previous_response_id")

    # Internal testing (TO BE DELETED)
    print("Previous id:", previous_id)

    # Get GPT response
    response = client.responses.create(
    model="gpt-4.1",
    previous_response_id = previous_id,
    input=[
        {"role": "system", "content": "You are a career mentor. Keep your answers short and encouraging."},
        {"role": "user", "content": user_message}
        ]
    )
    ai_response = response.output_text

    # Internal testing (TO BE DELETED)
    print("AI Response:", ai_response)

    session["previous_response_id"] = response.id

    # Convert GPT response to speech
    tts_response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="marin",
        input=ai_response
    )

    # Save TTS audio file
    audio_path = os.path.join("static", "audio", "response.mp3")
    with open(audio_path, "wb") as f:
        f.write(tts_response.read())

    # Internal testing (TO BE DELETED)
    print("TTS audio file saved:", audio_path)

    # 5️⃣ Return everything to frontend
    return jsonify({
        "message": "Success",
        "transcript": user_message,
        "ai_response": ai_response,
        "audio": audio_path
    })

