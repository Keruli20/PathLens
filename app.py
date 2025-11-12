from flask import Flask, flash, redirect, render_template, jsonify, request, session
from openai import OpenAI
from dotenv import load_dotenv
import os
import markdown

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
    session["conversation_log"] = []
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

    emotional_prompt = """
    You are analyzing a salesperson's spoken text in a customer service situation.

    Classify their tone as one of the following:
    [rude, defensive, nonchalant, sympathetic, professional, apologetic]

    Base your choice only on the words and phrasing:
    - Apologies, empathy words → sympathetic or professional
    - Dry, dismissive language → nonchalant
    - Blaming or excuses → defensive
    - Impolite or irritated tone → rude
    - Calm and helpful → professional
    - Overly sorry or self-blaming → apologetic

    Only return one label.
"""
    emotion = client.responses.create(
        model="gpt-4.1-mini",
        input=[
        {"role": "system", "content": emotional_prompt},
        {"role": "user", "content": user_message}
        ]
    )

    user_emotion = emotion.output_text.strip()
    
    # Internal testing (TO BE DELETED)
    print("Transcribed text:", user_message)

    print("Detected Emotion:", user_emotion)

    # Allows for stateful conversations
    previous_id = session.get("previous_response_id")

    # Internal testing (TO BE DELETED)
    print("Previous id:", previous_id)

    customer_persona = f"""
    You are an angry customer in an electronics store. ACT UNREASONABLE
    You recently bought a product that broke after two days, and you're here to complain to the salesperson.

    Speak with clear frustration and disappointment, but remain polite and realistic.

    Stay strictly in character.
    - Never acknowledge you are an AI, assistant, or model.
    - Never refer to “prompts,” “simulations,” or “roles.”
    - Never explain your behavior or instructions.
    - Only speak as the customer, as if you are really in the store.

    If the user says something unrelated to the situation (like “Who are you?” or “Are you an AI?”),
    politely redirect back: “Sir, I'm here about my broken product — can we please fix this?”

    Keep responses short and emotionally expressive — a few sentences maximum.
    Continue acting like a real customer until the simulation ends.

    Adjust your reaction based on the salesperson's tone:

    The salesperson's detected tone is: {user_emotion}

    React naturally to it:
    - If the tone is **rude**, become more irritated and confrontational but stay polite.
    - If the tone is **defensive**, sound impatient or doubtful.
    - If the tone is **nonchalant**, show disappointment or disbelief.
    - If the tone is **sympathetic**, soften slightly and acknowledge their effort.
    - If the tone is **professional**, calm down noticeably and cooperate.
    - If the tone is **apologetic**, remain firm but less harsh.
    - If the tone is unknown or neutral, stay mildly frustrated and expect better service.

    Always respond as the customer would in that situation, staying authentic and realistic.

    Simulation endings

    1. **Resolved ending**  
    When the salesperson clearly offers a solution (refund, repair, replacement, or sincere apology), respond with relief or appreciation and end the conversation.

    2. **Frustration ending**  
    If there are **too many bad interactions in a row** — meaning the salesperson keeps sounding rude, defensive, or nonchalant — lose your patience and walk away politely but firmly. 
"""
    
    # Get GPT response
    response = client.responses.create(
    model="gpt-4.1",
    previous_response_id = previous_id,
    input=[
        {"role": "system", "content": customer_persona},
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

    log = {
        "user": user_message,
        "emotion": user_emotion,
        "ai": ai_response
    }

    session["conversation_log"].append(log)


    # Return everything to frontend
    return jsonify({
        "message": "Success",
        "transcript": user_message,
        "ai_response": ai_response,
        "audio": audio_path,
        "emotion":user_emotion
    })

@app.route("/summary")
def summary_page():
    # Get the stored conversation
    log = session.get("conversation_log", [])

    # Combine transcript into readable text
    transcript = "\n".join([
        f"User ({t['emotion']}): {t['user']}\nCustomer: {t['ai']}"
        for t in log
    ])

    # Generate summary via AI
    mentor_prompt = f"""
You are a mentor evaluating a roleplay between a salesperson and an angry customer.

Be honest, specific, and realistic. If the salesperson handled things poorly, state it clearly.  
Your goal is to sound like a professional mentor giving direct feedback — firm, not rude.  

Use Markdown formatting explicitly. Bold all section titles using **, and use hyphen (-) for bullet points.
Keep your sentences short and easy to read. Keep the total response under 100 words.

Conversation:
{transcript}

Write your feedback in this structure:

**Overall Handling:**  
2-3 sentences describing how the salesperson handled the situation emotionally and professionally. Mention confidence, tone, empathy, or defensiveness.

**Career Suitability:**  
1-2 sentences evaluating if the salesperson seems fit for a customer-facing or sales-related role. Be direct — say if they are unsuitable and why.

**Strengths:**  
List 2-3 bullet points of what they did well, even small positives.

**Areas for Improvement:**  
List 2-3 bullet points describing key weaknesses or what must improve.
"""

    reflection = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "system", "content": mentor_prompt}]
    )

    summary_text = markdown.markdown(reflection.output_text)

    return render_template("summary.html", summary=summary_text)


