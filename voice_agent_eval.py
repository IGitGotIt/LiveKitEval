import asyncio
import time
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, elevenlabs
import speech_recognition as sr
from textblob import TextBlob
import pyttsx3
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, filename='latency_metrics.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Apple-like speech recognizer and TTS
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Semantic text conversion to reduce input size
def semantic_text_conversion(text):
    """Summarize text to reduce LLM input size by removing filler words."""
    blob = TextBlob(text)
    # Example: Simplify by extracting key nouns/verbs
    keywords = [word for word, pos in blob.tags if pos in ['NN', 'VB', 'VBP', 'VBZ']]
    simplified_text = ' '.join(keywords[:5])  # Limit to 5 keywords for brevity
    logging.info(f"Original: {text}, Simplified: {simplified_text}")
    return simplified_text if simplified_text else text

# Measure latency for a function
def measure_latency(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        latency = (time.time() - start_time) * 1000  # Convert to ms
        logging.info(f"{func.__name__} latency: {latency:.2f} ms")
        return result, latency
    return wrapper

@measure_latency
async def process_speech_to_text(audio):
    """Convert audio to text using speech recognition."""
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio_data)  # Placeholder for Apple Speech
            simplified_text = semantic_text_conversion(text)
            return simplified_text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"

@measure_latency
async def process_llm(text):
    """Process text with LLM (simulated with simple response)."""
    # Placeholder for OpenAI or Apple LLM; here we echo simplified input
    response = f"Response to: {text}"
    return response

@measure_latency
async def process_text_to_speech(text):
    """Convert text to speech."""
    tts_engine.save_to_file(text, 'output.wav')
    tts_engine.runAndWait()
    return 'output.wav'

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    agent = Agent(
        instructions="You are a voice assistant measuring WebRTC latency.",
    )
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),  # Fallback STT
        tts=elevenlabs.TTS(),  # Fallback TTS
    )

    # Measure end-to-end latency
    start_e2e = time.time()
    audio = None  # Simulate audio input
    text, stt_latency = await process_speech_to_text(audio)
    response, llm_latency = await process_llm(text)
    audio_output, tts_latency = await process_text_to_speech(response)
    
    e2e_latency = (time.time() - start_e2e) * 1000  # Convert to ms
    logging.info(f"End-to-end latency: {e2e_latency:.2f} ms")
    
    # Log latency savings from semantic conversion
    original_text = "Sample user input with filler words like um and uh"
    simplified = semantic_text_conversion(original_text)
    savings = len(original_text) / (len(simplified) + 1) * 100
    logging.info(f"Semantic conversion reduced input size by {savings:.2f}%")
    
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions=f"Processed: {response}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
