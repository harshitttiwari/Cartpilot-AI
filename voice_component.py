# voice_component.py
import re
import warnings
import streamlit as st
import streamlit.components.v1 as components

# Languages for Speech-to-Text (STT) and Voice Synthesis
SUPPORTED_LANGUAGES = {
    "English (India) 🇮🇳": "en-IN",
    "English (US) 🇺🇸": "en-US",
    "Hindi (हिंदी) 🇮🇳": "hi-IN",

}

def render_voice_controller():
    """
    Renders an interactive Voice Command Controller above the chat interface.
    Features:
    - Real-time Web Speech API Microphone recorder
    - Multi-language Speech Recognition (English, Hindi, Spanish, etc.)
    - Visual pulse animation during listening state
    - TTS (Text-to-Speech) Audio toggle switch
    - Automatic injection into the chat prompt
    """
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True

    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True

    if "selected_voice_lang" not in st.session_state:
        st.session_state.selected_voice_lang = "en-IN"

    with st.container():
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.6, 1.3, 0.8])

        with ctrl_col1:
            lang_label = st.selectbox(
                "🌐 Language",
                options=list(SUPPORTED_LANGUAGES.keys()),
                index=0,
                key="voice_language_select",
                label_visibility="collapsed",
                help="Select your spoken language for voice recognition",
            )
            lang_code = SUPPORTED_LANGUAGES[lang_label]
            st.session_state.selected_voice_lang = lang_code

        with ctrl_col2:
            st.session_state.tts_enabled = st.checkbox(
                "🔊 Read Aloud (TTS)",
                value=st.session_state.tts_enabled,
                help="Automatically speak bot confirmations and smart suggestions",
            )

        with ctrl_col3:
            st.markdown("<div style='text-align:right;padding-top:4px;'><span style='color:#00e676;font-size:0.8rem;font-weight:600;'>🟢 Mic Ready</span></div>", unsafe_allow_html=True)

        # Web Speech API Controller Component
        mic_component_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: transparent;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 2px 0;
                }}
                .mic-button {{
                    background: linear-gradient(135deg, #FF4B4B 0%, #FF8533 100%);
                    color: white;
                    border: none;
                    border-radius: 20px;
                    padding: 6px 16px;
                    font-size: 12.5px;
                    font-weight: 600;
                    cursor: pointer;
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    box-shadow: 0 2px 10px rgba(255, 75, 75, 0.3);
                    transition: all 0.2s ease;
                    outline: none;
                }}
                .mic-button:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 4px 14px rgba(255, 75, 75, 0.45);
                }}
                .mic-button.listening {{
                    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                    animation: pulse 1.5s infinite;
                    box-shadow: 0 0 16px rgba(231, 76, 60, 0.8);
                }}
                @keyframes pulse {{
                    0% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.04); }}
                    100% {{ transform: scale(1); }}
                }}
                .status-badge {{
                    font-size: 12px;
                    color: #999;
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                }}
                .recording-indicator {{
                    display: none;
                    width: 7px;
                    height: 7px;
                    background-color: #ff3333;
                    border-radius: 50%;
                    animation: blink 1s infinite;
                }}
                @keyframes blink {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.2; }}
                }}
            </style>
        </head>
        <body>
            <button id="micBtn" class="mic-button" onclick="handleMicClick()">
                <span id="micIcon">🎙️</span>
                <span id="btnText">Speak</span>
            </button>
            <div class="status-badge">
                <span id="recDot" class="recording-indicator"></span>
                <span id="statusLabel">Click mic or type grocery commands</span>
            </div>

            <script>
                let recognition = null;
                let isListening = false;
                const langCode = "{lang_code}";

                if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {{
                    const SpeechRecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SpeechRecognitionClass();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    recognition.lang = langCode;

                    recognition.onstart = function() {{
                        isListening = true;
                        document.getElementById('micBtn').classList.add('listening');
                        document.getElementById('btnText').innerText = 'Listening...';
                        document.getElementById('recDot').style.display = 'inline-block';
                        document.getElementById('statusLabel').innerText = 'Listening in {lang_label}...';
                    }};

                    recognition.onresult = function(event) {{
                        const transcript = event.results[0][0].transcript.trim();
                        document.getElementById('statusLabel').innerText = 'Captured: "' + transcript + '"';
                        
                        // Automatically inject transcript into Streamlit's Chat Input textarea
                        try {{
                            const parentDoc = window.parent.document;
                            const chatTextarea = parentDoc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                            if (chatTextarea) {{
                                const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, "value")?.set || 
                                                     Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
                                
                                if (nativeSetter) {{
                                    nativeSetter.call(chatTextarea, transcript);
                                }} else {{
                                    chatTextarea.value = transcript;
                                }}
                                
                                chatTextarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                chatTextarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                
                                // Auto-trigger submission
                                setTimeout(() => {{
                                    const sendBtn = parentDoc.querySelector('button[data-testid="stChatInputSubmitButton"]');
                                    if (sendBtn) {{
                                        sendBtn.removeAttribute('disabled');
                                        sendBtn.click();
                                    }}
                                    chatTextarea.dispatchEvent(new KeyboardEvent('keydown', {{
                                        key: 'Enter',
                                        code: 'Enter',
                                        keyCode: 13,
                                        which: 13,
                                        bubbles: true,
                                        cancelable: true
                                    }}));
                                }}, 350);
                            }}
                        }} catch (err) {{
                            console.log("Could not auto-submit to chat input:", err);
                        }}
                    }};

                    recognition.onerror = function(event) {{
                        isListening = false;
                        resetMicUI();
                        if (event.error === 'not-allowed') {{
                            document.getElementById('statusLabel').innerText = '❌ Microphone access denied. Allow mic permissions in browser.';
                        }} else if (event.error === 'no-speech') {{
                            document.getElementById('statusLabel').innerText = '⚠️ No speech detected. Please try again.';
                        }} else {{
                            document.getElementById('statusLabel').innerText = '⚠️ Voice error: ' + event.error;
                        }}
                    }};

                    recognition.onend = function() {{
                        isListening = false;
                        resetMicUI();
                    }};
                }} else {{
                    document.getElementById('statusLabel').innerText = '⚠️ Speech Recognition API not supported in this browser. Please use Chrome/Edge.';
                }}

                function resetMicUI() {{
                    document.getElementById('micBtn').classList.remove('listening');
                    document.getElementById('btnText').innerText = 'Speak Command';
                    document.getElementById('recDot').style.display = 'none';
                }}

                function handleMicClick() {{
                    if (!recognition) return;
                    if (isListening) {{
                        recognition.stop();
                    }} else {{
                        recognition.lang = '{lang_code}';
                        try {{
                            recognition.start();
                        }} catch(e) {{
                            console.log("Recognition start error:", e);
                        }}
                    }}
                }}
            </script>
        </body>
        </html>
        """
        _safe_html_render(mic_component_html, height=44)


def _safe_html_render(html_content: str, height: int = 0):
    """Safe HTML / component renderer across Streamlit versions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        components.html(html_content, height=height)


def render_tts_speaker(text_to_speak: str):
    """
    Speaks the assistant's confirmation and smart suggestions aloud using browser Text-to-Speech.
    Automatically strips markdown markup, links, and emojis for natural pronunciation.
    """
    if not text_to_speak or not st.session_state.get("tts_enabled", True):
        return

    # Clean text of markdown bullet points, bold markers, emojis, and code backticks
    clean_text = re.sub(r"[*_`#~]", "", str(text_to_speak))
    clean_text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", clean_text)  # Strip markdown links
    clean_text = re.sub(r"[🤖👤🛒💡🔊🎙️•\-–]", " ", clean_text)    # Strip common UI symbols
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    
    # Escape quotes and backslashes for safe JavaScript injection
    js_safe_text = clean_text.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

    # Pick speech synthesis language code based on session state
    tts_lang = st.session_state.get("selected_voice_lang", "en-US")

    tts_html = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <script>
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel(); // Stop any pending speech
                const utterance = new SpeechSynthesisUtterance("{js_safe_text}");
                utterance.lang = "{tts_lang}";
                utterance.rate = 1.05;
                utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            }}
        </script>
    </body>
    </html>
    """
    _safe_html_render(tts_html, height=0)
