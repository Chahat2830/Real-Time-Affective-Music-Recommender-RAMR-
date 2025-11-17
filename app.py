import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from deepface import DeepFace
import cv2

# Suppress warnings that are safe to ignore in this context
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL CONFIGURATION ---

# 1. RTC Configuration Fix: Use a comprehensive list of public STUN servers.
# NOTE: If this configuration fails, a dedicated TURN server with credentials is required
# because the deployment environment's firewall is blocking UDP traffic.
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        {"urls": ["stun:stunserver.org:3478"]},
        # Adding more public servers for redundancy
        {"urls": ["stun:stun.services.mozilla.com"]},
        {"urls": ["stun:stun.nextcloud.com:443"]},
    ]
}

# --- 1. SETUP AND LOAD ARTIFACTS ---

@st.cache_resource
def load_artifacts():
    """Loads the trained model, scaler, and cleaned DataFrame."""
    try:
        # Load the final components saved from the consolidated script
        knn_model = joblib.load('knn_mood_model_final.joblib')
        scaler = joblib.load('feature_scaler_final.joblib')
        # Load the DataFrame with the corrected (reset) index
        df = pd.read_csv('processed_songs_mood_final.csv', index_col=0)
        
        # Define the exact list of features used for training
        feature_cols = [
            'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
            'key', 'loudness', 'mode', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
            'nlp_mood_score'
        ]
        
        # Dictionary to map detected emotions to a numerical mood score (Valence)
        emotion_to_valence = {
            'happy': 0.9,
            'sad': 0.1,
            'surprise': 0.7,
            'angry': 0.2,
            'fear': 0.3,
            'neutral': 0.5,
            'disgust': 0.2
        }
        
        return knn_model, scaler, df, feature_cols, emotion_to_valence
        
    except FileNotFoundError as e:
        st.error(f"Error loading required file: {e}. Please ensure model files ('_final.joblib', etc.) are in this directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during artifact loading: {e}")
        st.stop()

# Load everything once
try:
    knn_mood, scaler, df, FEATURE_COLUMNS, EMOTION_TO_VALENCE = load_artifacts()
except Exception:
    # Stop execution if artifacts failed to load
    st.stop()


# --- 2. CORE RECOMMENDATION LOGIC ---

def get_recommendations_from_features(input_features, df_songs, knn_model, scaler, feature_cols, n_recommendations=7):
    """
    Recommends songs based on a dictionary of feature values using k-Nearest Neighbors.
    """
    
    # 1. Convert input features (dictionary) to a DataFrame row
    input_df = pd.DataFrame([input_features])
    
    # 2. Select and reshape raw features
    raw_features = input_df[feature_cols].values.reshape(1, -1)
    
    # 3. Scale the input features
    scaled_input = scaler.transform(raw_features)
    
    # 4. Find the nearest neighbors (using the trained model)
    distances, indices = knn_model.kneighbors(scaled_input, n_neighbors=n_recommendations + 1)
    
    # 5. Extract results (We skip index 0 in case the input was from the dataset itself)
    recommended_indices = indices[0][1:] 
    recommended_songs_df = df_songs.loc[recommended_indices]
    
    results = [
        f"**{row['track_name']}** by {row['artists'].replace(';', ', ')}" 
        for i, row in recommended_songs_df.iterrows()
    ]
    return results

# --- 3. REAL-TIME VIDEO TRANSFORMER CLASS ---

class EmotionDetector(VideoTransformerBase):
    """
    A class to capture video frames, detect faces, and analyze emotions 
    in real-time using DeepFace and OpenCV.
    """
    def __init__(self):
        self.emotion = "neutral"  # Initialize with a lower-case default emotion
        # Use Haar Cascade for fast, simple face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Frame counter for performance optimization
        self.frame_count = 0
        self.SKIP_FRAMES = 5  # Analyze only once every 5 frames

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Optimization: Only process DeepFace every few frames
        self.frame_count += 1
        if self.frame_count % self.SKIP_FRAMES == 0:
            if self.frame_count >= 10000: self.frame_count = 0 # Simple counter reset

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                (x, y, w, h) = faces[0]  # Analyze the first detected face
                
                # Crop and analyze the face region
                face_img = img[y:y+h, x:x+w]
                
                # DeepFace analysis
                try:
                    # Enforce detection=False, silent=True for deployment stability
                    result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                    
                    if isinstance(result, list) and len(result) > 0:
                        self.emotion = result[0]['dominant_emotion']
                        
                except Exception:
                    # Catch internal DeepFace errors or face detection failures
                    self.emotion = "Processing..."
            
        # Draw a rectangle and put text on the frame using the latest known position/emotion
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Mood: {self.emotion.upper()}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Display emotion in a fixed spot if no face is detected
            cv2.putText(img, f"Mood: {self.emotion.upper()} (No Face)", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        return img

# --- 4. STREAMLIT UI LAYOUT ---

st.set_page_config(page_title="Mood-Based Song Recommender", layout="wide")
st.title("🎧 Real-Time Mood-Based Song Recommender")
st.markdown("This application uses your webcam to detect your dominant emotion and recommends songs with matching audio features.")

# CRITICAL DEPLOYMENT WARNING
st.warning("""
    **Deployment Error Alert:** The traceback you are seeing (`AttributeError: 'NoneType' object has no attribute 'is_alive'`) 
    is a known bug in older versions of the `streamlit-webrtc` dependency library, and **cannot be fixed in app.py.**
    To resolve this, please ensure your `requirements.txt` file specifies the latest version of `streamlit-webrtc` (e.g., `>=0.49.0`) 
    and redeploy your application to force an update.
""")

tab1, tab2 = st.tabs(["📸 Live Mood Detection", "🔎 Search by Song"])

# --- TAB 1: LIVE MOOD DETECTION ---
with tab1:
    st.header("Live Mood to Music Match")
    st.markdown("1. Allow camera access and look at the screen.")
    st.markdown("2. Once the dominant mood stabilizes, press the button below.")

    # Start the webcam stream using the custom transformer and robust RTC config
    ctx = webrtc_streamer(
        key="emotion_detector_live",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=EmotionDetector,
        rtc_configuration=RTC_CONFIGURATION
    )
    
    # Defensive Fix: Check if ctx and ctx.state exist before checking the connection state
    is_failed = False
    if ctx and hasattr(ctx, 'state') and hasattr(ctx.state, 'ice_connection_state'):
        if ctx.state.ice_connection_state == "failed":
            is_failed = True

    if is_failed:
        st.error("Connection failed! Your network/deployment environment may be blocking UDP ports. Consider using a dedicated TURN server (requires credentials).")

    if ctx.video_transformer:
        # Retrieve the emotion property safely
        current_emotion = ctx.video_transformer.emotion.lower()
        
        st.subheader(f"Mood Tracker: **{current_emotion.upper()}**")

        if st.button("Generate Mood-Matched Playlist 🎵"):
            
            if current_emotion in EMOTION_TO_VALENCE:
                with st.spinner(f"Generating songs for mood: {current_emotion.upper()}..."):
                    
                    # 1. Get the required valence score
                    detected_mood_valence = EMOTION_TO_VALENCE[current_emotion]
                    
                    # 2. Get average features of the dataset as a baseline
                    base_features = df[FEATURE_COLUMNS].mean().to_dict()
                    
                    # 3. Inject the desired features based on the detected mood:
                    base_features['valence'] = detected_mood_valence 
                    base_features['nlp_mood_score'] = detected_mood_valence
                    
                    # Custom feature adjustments to make 'Happy' songs more energetic
                    if current_emotion == 'happy':
                        base_features['danceability'] = 0.85 
                        base_features['energy'] = 0.8
                    elif current_emotion == 'sad':
                        base_features['danceability'] = 0.3
                        base_features['energy'] = 0.2
                        
                    # 4. Get recommendations
                    recommendations = get_recommendations_from_features(
                        input_features=base_features, 
                        df_songs=df, 
                        knn_model=knn_mood, 
                        scaler=scaler,
                        feature_cols=FEATURE_COLUMNS,
                        n_recommendations=7
                    )

                    st.success(f"Generated playlist for {current_emotion.upper()} mood!")
                    for i, song in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {song}")
            else:
                st.warning(f"Mood '{current_emotion.upper()}' is stabilizing. Please wait for a clear mood reading.")

# --- TAB 2: SEARCH BY SONG ---
with tab2:
    st.header("Search for a Song & Find Similar Tracks")
    st.info("This mode uses a song's existing audio/mood features to find similar tracks.")
    
    # Create the search index column if it doesn't exist (useful for robustness)
    if 'track_search' not in df.columns:
        df['track_search'] = df['track_name'] + ' by ' + df['artists'].str.replace(';', ', ', regex=False)

    search_term = st.text_input("Enter Song Name or Artist (e.g., Hold On, Ed Sheeran)", 
                                 key="search_term_input")
    
    if search_term:
        match = df[df['track_search'].str.contains(search_term, case=False, na=False)]
        
        if match.empty:
            st.warning(f"No songs found matching **'{search_term}'**.")
        else:
            # Limit display to avoid massive select box
            display_matches = match['track_search'].head(50).tolist()
            selected_song_option = st.selectbox("Select a Song:", display_matches)
            
            if st.button("Get Recommendations (Search Mode)", key="search_button"):
                with st.spinner('Generating recommendations...'):
                    # Find the original song row using the selected option
                    input_song = df[df['track_search'] == selected_song_option].iloc[0]
                    input_features = input_song[FEATURE_COLUMNS].to_dict()
                    
                    # Get recommendations
                    recommendations = get_recommendations_from_features(
                        input_features=input_features, 
                        df_songs=df, 
                        knn_model=knn_mood, 
                        scaler=scaler,
                        feature_cols=FEATURE_COLUMNS
                    )
                    
                    st.success(f"Playlist based on {input_song['track_name']} generated!")
                    # Skip the first recommendation as it will be the song itself due to KNN
                    for i, song in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {song}")
