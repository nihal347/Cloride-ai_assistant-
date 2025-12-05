from os import system
import speech_recognition as sr
from playsound import playsound
from gpt4all import GPT4All
import sys 
import whisper
import warnings
import time
import os 

wake_word = "cloride"

model =GPT4All("F:/projects-dev/assistant-xoxo/llm-model/gpt4all-falcon-newbpe-q4_0.gguf", backend="gptj", n_threads=4, allow_download=False)

r = sr.Recognizer()

tiny_model_path = os.path.expanduser(r"C:\Users\nihal akndo\.cache\whisper\tiny.pt")
base_model_path = os.path.expanduser(r"C:\Users\nihal akndo\.cache\whisper\base.pt")

tiny_model = whisper.load_model(tiny_model_path, device="cpu")
base_model = whisper.load_model(base_model_path, device="cpu")
listening_for_wake_word = True
source = sr.Microphone()
if sys.platform != 'darwin':
    import pyttsx3
    engine = pyttsx3.init()

def speak(text):
    if sys.platform == 'darwin':
        ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+/ ")
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        system(f"say '{clean_text}'")
    else:
        engine.say(text)
        engine.runAndWait()

def listen_for_wake_word(audio):
    global listening_for_wake_word
    with open("wake_detect.wav", "wb") as f:
        f.write(audio.get_wav_data())
    result = tiny_model.transcribe("wake_detect.wav")
    text_input = result["text"]
    if text_input.lower().strip() == wake_word:
        print("wake word detected. Please speak.")
        speak('listening...')
        listening_for_wake_word = False

def prompt_gpt(audio):
    global listening_for_wake_word
    try:
        with open("prompt.wave", "wb") as f:
            f.write(audio.get_wav_data())
        result = base_model.transcribe("prompt.wav")
        prompt_text = result["text"]
        if len(prompt_text.strip()) == 0: 
            print("no prompt detected. Please speak.")
            speak('no prompt detected. Please speak.')
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)
            output = model.generate(prompt_text, max_tokens=200)
            print('Cloride: ' + output)
            speak(output)
            print('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = True
    except Exception as e:
        print("Prompt error:", e)

def callback(recognizer, audio):
    global listening_for_wake_word
    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)
def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
        print('Say', wake_word, 'to wake me up. \n')
        r.listen_in_background(s, callback)
        while true:
            time.sleep(1)

if __name__ == "__main__":
    start_listening()