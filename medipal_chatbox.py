import os
import gradio as gr
import datetime
from gtts import gTTS
from faster_whisper import WhisperModel
import requests
from pathlib import Path

url = "http://127.0.0.1:30000/ask" # medipal is running under the api

logo_path = Path(__file__).parent/"assets/screenshots/logo_small.PNG"

workspace_base_path = os.getcwd()
audio_path = os.path.join(workspace_base_path, "assets", "audio") 
out_path = Path(audio_path)
out_path.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
asr_model = WhisperModel("turbo")

def transcribe_audio(audio_file):
    """
    convert a audio file to text(ASR)    
    """
    if not audio_file:
        return None       
    segments, info  = asr_model.transcribe(audio_file, beam_size=5)
    text = "".join(seg.text for seg in segments)
    return text

def generate_response(query: str):
    """
    Call medipal api to generate the answer.   
    """
    if not query:
        return "I didn’t catch anything. Please try speaking or typing."
      
    payload = {"query": query}

    # Send POST request
    response = requests.post(url, json=payload)    
    ai_message = None
    if response.status_code == 200:
        ai_message = response.json()["message"]        
    else:
        print("Error:", response.status_code, response.text)
    return ai_message

def synthesize_audio(text):
    """
    Convert text to audio(TTS)    
    """
    filename = f"Response-{timestamp}.wav"    
    file_path = os.path.join(audio_path, filename) 
    # Text to Audio
    tts = gTTS(text, lang='en')
    tts.save(file_path)
    return file_path    

# ----------------------------
# Core interaction function
# ----------------------------
def voice_assistant(audio, text, history_state, use_audio_first):
    """
    audio: (sr, np.ndarray) or None
    text: str or None
    history_state: list[tuple[str, str]]
    use_audio_first: bool – if True, prefer audio when both provided
    """
    try:
        # Decide which input to use
        chosen_text = None
        if use_audio_first and audio is not None:
            chosen_text = transcribe_audio(audio)
        elif text:
            chosen_text = text
        elif audio is not None:
            chosen_text = transcribe_audio(audio)

        if not chosen_text:            
            raise gr.Error("Please provide either a voice recording or a text prompt.")

        query = chosen_text.strip()        

        # LLM response
        reply = generate_response(query)

        # Optional TTS
        audio_out = synthesize_audio(reply)

        # Update history for Chatbot
        history_state = history_state or []
        history_state.append((query, reply))

        # Return: chatbot, text output, audio output, cleared text box, preserved history
        return history_state, reply, audio_out, gr.update(value=""), history_state

    except Exception as e:
        #logging.exception("Error in voice_assistant")
        # Show the error gracefully in the text output; leave history unchanged
        return history_state, f"Error: {e}", None, gr.update(), history_state

def clear_history():
    return [], gr.update(value="")

# ----------------------------
# Custom CSS (sleek look)
# ----------------------------
CSS = """
    .gradio-container {max-width: 1024px !important;}
    #title {
    text-align: center;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    }
    #subtitle {
    text-align: center;
    color: #6b7280;
    margin-top: -10px;
    margin-bottom: 12px;
    }
    .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.75) 0%, rgba(250,250,250,0.75) 100%);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.06);
    }
    .footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.875rem;
    margin-top: 8px;
    }
    """

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft(), css=CSS, fill_height=True) as demo:
    gr.HTML('<div id="title">MediPal</div>')
    gr.HTML('<div id="subtitle">Your AI friend for medical and clinical Q&A</div>')

#     gr.HTML(f"""<div style="display: flex; align-items: flex-start;">
#   <div style="flex: 0 0 150px;">
#     <img src="https://github.com/Montreal-JieZhu/MediPal/blob/main/assets/screenshots/logo_small.PNG" alt="Logo" width="140" height="140" style="margin-right:15px;" />
#   </div>
#   <div style="flex: 1; padding-left: 15px;">
#     <p>
#       <b>MediPal</b> is your AI friend for medical and clinical Q&A. 
#       <ul>
#          <li>It is an open-source medical assistant that provides comprehensive mediciation information and symptom-based recommendations using natural-language understanding which supports voice and message conversation.</li>
#          <li>Powered by local Huggingface LLMs, embedding model, cross-encoder(BERT) and whisper with AI Agent development frameworks like langchain, langgraph and many tools. </li>
#       </ul>
#     </p>
#   </div>
# </div>""")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### Input")
                use_audio_first = gr.Checkbox(
                    value=True, label="Prefer voice if both provided"
                )
                mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Record your prompt",
                    waveform_options={"show_controls": True},
                )
                text_in = gr.Textbox(
                    label="⌨️ Or type here",
                    placeholder="Ask a question or say something...",
                    lines=2
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear History")

        with gr.Column(scale=3):
            with gr.Group(elem_classes="card"):
                gr.Markdown("### Conversation")
                chat = gr.Chatbot(
                    label="History",
                    height=420,
                    show_copy_button=True,
                    avatar_images=(None, None),  # plug paths if you want custom avatars
                    bubble_full_width=False,
                )

            with gr.Group(elem_classes="card"):
                gr.Markdown("### Assistant Outputs")
                out_text = gr.Textbox(
                    label="Assistant (text output)", lines=4, interactive=False
                )
                out_audio = gr.Audio(
                    label="Assistant (voice output)", autoplay=True, interactive=False
                )

    history_state = gr.State([])

    # Click -> process
    send.click(
        voice_assistant,
        inputs=[mic, text_in, history_state, use_audio_first],
        outputs=[chat, out_text, out_audio, text_in, history_state],
        queue=True,
        show_progress=True
    )

    # Enter/submit on the textbox also triggers send
    text_in.submit(
        voice_assistant,
        inputs=[mic, text_in, history_state, use_audio_first],
        outputs=[chat, out_text, out_audio, text_in, history_state],
        queue=True,
        show_progress=True
    )

    # Clear
    clear_btn.click(
        clear_history,
        inputs=None,
        outputs=[chat, text_in],
    )
        
def launch_chatbox():
    demo.launch(server_port=30001) # I hard code the port here, you can change it

__all__ = ["launch_chatbox"]

if __name__ == "__main__":    
    launch_chatbox()

# python medipal_chatbox.py