import cv2
import speech_recognition as sr
from deepface import DeepFace
import language_tool_python
from collections import defaultdict, Counter
import threading
import time
import streamlit as st

# Global variables
emotion_count = defaultdict(int)
current_emotions = []

# Function to capture facial expressions
def capture_facial_expressions():
    global current_emotions
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Camera not responding or in use.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Frame not captured.")
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                dominant_emotion = analysis[0]['dominant_emotion']
            else:
                dominant_emotion = analysis['dominant_emotion']
            current_emotions.append(dominant_emotion)
        except Exception as e:
            st.error(f"Error in detecting expression: {str(e)}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to get the most frequent emotion
def get_dominant_emotion():
    if current_emotions:
        emotion_counter = Counter(current_emotions)
        dominant_emotion = emotion_counter.most_common(1)[0][0]
        st.write(f"Dominant emotion after question: {dominant_emotion}")
        emotion_count[dominant_emotion] += 1
        current_emotions.clear()
        return dominant_emotion
    return "No emotion detected"

# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak into the microphone...")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            st.write(f"Recognized Text: {text}")
            return text
        except sr.WaitTimeoutError:
            st.error("No response detected within 10 seconds.")
            return ""
        except sr.UnknownValueError:
            st.error("Sorry, I did not understand the audio.")
            return ""
        except sr.RequestError:
            st.error("Error with the speech recognition service.")
            return ""

# Function to evaluate keywords
def evaluate_keywords(answer, keywords):
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    return (keyword_count / len(keywords)) * 100

# Function to check grammar
def grammar_check(answer):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(answer)
    errors = len(matches)
    total_words = len(answer.split())
    accuracy = max(0, (total_words - errors) / total_words * 100)
    return accuracy

# Function to ask questions
def ask_questions(questions_keywords):
    answers = []
    for question, keywords in questions_keywords.items():
        st.write(f"\n**Question:** {question}")

        # Reset emotions
        current_emotions.clear()

        # Start facial expression capture in a separate thread
        facial_thread = threading.Thread(target=capture_facial_expressions)
        facial_thread.start()

        # Get the answer from speech
        answer = speech_to_text()
        answers.append(answer)

        # Stop facial expression capture
        facial_thread.join()

        # Analyze emotions
        dominant_emotion = get_dominant_emotion()

        if answer:
            # Keyword score
            keyword_score = evaluate_keywords(answer, keywords)
            st.write(f"**Keyword Match Score:** {keyword_score:.2f}")

            # Grammar score
            grammar_score = grammar_check(answer)
            st.write(f"**Grammar Accuracy Score:** {grammar_score:.2f}")

    return answers

# Function to generate a final emotion report
def generate_final_emotion_report():
    total_emotions = sum(emotion_count.values())
    if total_emotions == 0:
        st.write("No emotions were detected during the interview.")
        return

    st.write("\n### Final Emotion Report")
    for emotion, count in emotion_count.items():
        emotion_percentage = (count / total_emotions) * 100
        st.write(f"{emotion.capitalize()}: {emotion_percentage:.2f}%")

# Main function
def main():
    st.title("AI-Powered Interview Bot")

    questions_keywords = {
        "Tell me about yourself.": ["experience", "skills", "background"],
        "Why do you want to work here?": ["company", "values", "mission"],
        "What are your strengths?": ["strength", "skill", "expertise"],
        "Where do you see yourself in 5 years?": ["future", "goals", "career"],
        "Tell me about a challenge you've faced.": ["challenge", "overcome", "problem"],
    }

    if st.button("Start Interview"):
        answers = ask_questions(questions_keywords)
        st.write("\n### Answers")
        for i, ans in enumerate(answers, start=1):
            st.write(f"**Answer {i}:** {ans}")
        generate_final_emotion_report()

if __name__ == "__main__":
    main()
