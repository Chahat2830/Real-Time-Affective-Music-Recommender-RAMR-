import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import cv2
import av
import logging
from deepface import DeepFace

# Set up logging for debugging issues, especially with video/threading
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Caching DeepFace Models for faster loading ---
@st.cache_resource
def load_deepface_model():
    """Load the emotion model once and cache it using st.cache_resource."""
    try:
        st.info("Initializing DeepFace models for emotion analysis (first run may take a moment)...")
        # Force a small analysis call to trigger model download/loading outside the video loop
        # This uses VGG-Face detector and the emotion model
        _ = DeepFace.analyze(np.zeros((48, 48, 3), dtype='uint8'), actions=['emotion'], enforce_detection=False, silent=True)
        st.success("DeepFace models loaded successfully!")
    except Exception as e:
        st.error(f"Error during DeepFace model pre-load: {e}")
        logger.error(f"DeepFace pre-load error: {e}")
        
# Pre-load models before the main application runs
load_deepface_model()


# --- Configuration ---
st.set_page_config(layout="centered", page_title="Real-Time Affective Analyzer")

# --- DeepFace Video Processor ---
class DeepfaceVideoProcessor(VideoProcessorBase):
    """
    Processes video frames in real-time to analyze facial emotions using DeepFace.
    """
    def __init__(self):
        self.emotion = "Neutral"
        self.frame_count = 0
        # DeepFace is heavy, so we process only every 5th frame to maintain low latency
        self.skip_frames = 5 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.skip_frames == 0:
            # Convert to RGB for DeepFace analysis
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- Emotion Analysis (DeepFace) ---
            try:
                analysis = DeepFace.analyze(
                    img_path=img_rgb, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv', # Fast detection backend
                    silent=True
                )

                if analysis and isinstance(analysis, list) and analysis[0].get('dominant_emotion'):
                    self.emotion = analysis[0]['dominant_emotion'].capitalize()
                else:
                    self.emotion = "No Face Detected"

            except Exception as e:
                # Catching analysis errors without crashing the main thread
                self.emotion = "Error Analyzing"
                logger.error(f"DeepFace analysis failed: {e}")


        # --- Drawing Output ---
        # Draw the dominant emotion on the BGR frame (img)
        text = f"Emotion: {self.emotion}"
        cv2.putText(
            img, 
            text, 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), # Green color
            2, 
            cv2.LINE_AA
        )
        
        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---

st.title("🎭 Real-Time Affective Analyzer")
st.markdown("""
This application uses `streamlit-webrtc` and `DeepFace` for real-time emotion recognition.
*Note: You may encounter runtime errors (like `AttributeError: 'NoneType' object has no attribute 'is_alive'`) if the video session is stopped abruptly, as this is a known threading bug in the `streamlit-webrtc` library.*
""")

# --- WebRTC Streamer ---

# The key helps maintain session state across reruns.
webrtc_ctx = webrtc_streamer(
    key="deepface-analyzer-v2", # Changed key slightly
    video_processor_factory=DeepfaceVideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# --- Emotion Display and Music Placeholder ---

if webrtc_ctx.state.playing:
    # Access the latest emotion result from the video processor state
    if webrtc_ctx.video_processor:
        st.subheader("Analysis Result:")
        st.info(f"Dominant Emotion: **{webrtc_ctx.video_processor.emotion}**")
        
        # --- Affective Music Recommender Placeholder ---
        emotion_map = {
            "Happy": "Upbeat Pop",
            "Sad": "Mellow Acoustic",
            "Angry": "Aggressive Rock",
            "Fear": "Ambient Instrumental",
            "Disgust": "Experimental Jazz",
            "Surprise": "Energetic Electronic",
            "Neutral": "Calm Classical",
            "No Face Detected": "Awaiting Input",
            "Error Analyzing": "Please Restart Stream"
        }
        
        current_emotion = webrtc_ctx.video_processor.emotion
        suggested_music = emotion_map.get(current_emotion, "No Suggestion")
        
        st.subheader("🎵 Music Recommendation:")
        st.success(f"Based on your current emotion, we suggest: **{suggested_music}**")
        
        st.markdown("---")
        st.caption("Hint: We are only analyzing every 5th frame to maintain low latency.")
else:
    st.warning("Click 'Start' to enable the webcam and begin real-time emotion analysis.")
