import cv2
import mediapipe as mp
import streamlit as st
from compound_db import COMPOUND_PROPERTIES
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe Hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Gesture Mapping based on finger count
GESTURE_MAPPING = {
    1: 'Hydrogen (H)',   # 1 finger
    2: 'Oxygen (O)',     # 2 fingers
    3: 'Sodium (Na)',    # 3 fingers
    4: 'Chlorine (Cl)',  # 4 fingers
    5: 'Carbon (C)',     # 5 fingers
}

# To store selected elements
selected_elements = []

# Initialize LangChain LLM (Using OpenAI for this example)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# Function to count extended fingers
def count_extended_fingers(landmarks):
    """Count the number of extended fingers based on the hand landmarks."""
    extended_fingers = 0
    if landmarks[4].y < landmarks[3].y:  # Thumb up
        extended_fingers += 1
    if landmarks[8].y < landmarks[7].y:  # Index finger up
        extended_fingers += 1
    if landmarks[12].y < landmarks[11].y:  # Middle finger up
        extended_fingers += 1
    if landmarks[16].y < landmarks[15].y:  # Ring finger up
        extended_fingers += 1
    if landmarks[20].y < landmarks[19].y:  # Pinky finger up
        extended_fingers += 1
    return extended_fingers

# Function to combine elements into compounds
def combine_elements(elements):
    """Combine elements into a compound (based on specific gestures)."""
    if 'Hydrogen (H)' in elements and 'Oxygen (O)' in elements:
        return 'Water (H₂O)'  # Combine H + O to form H₂O
    elif 'Sodium (Na)' in elements and 'Chlorine (Cl)' in elements:
        return 'Sodium Chloride (NaCl)'  # Combine Na + Cl to form NaCl
    elif 'Carbon (C)' in elements and 'Oxygen (O)' in elements:
        return 'Carbon Dioxide (CO₂)'  # Combine C + O to form CO₂
    elif 'Carbon (C)' in elements and 'Hydrogen (H)' in elements:
        return 'Methane (CH₄)'  # Combine C + H to form CH₄
    else:
        return None  # Invalid combination

# Streamlit App
st.set_page_config(
    page_title="Chemistry Learning Tool",  # Page title
    page_icon="⚗️",  # Page favicon
    layout="wide"  # Wide layout for more space
)

# Title with some styling
st.markdown("""
    <h1 style="text-align: center; color: #4CAF50;">⚗️ Chemistry Learning Tool🧪</h1>
    <p style="text-align: center; color: #555;">Create compounds by gestures and explore their properties🧪</p>
    """, unsafe_allow_html=True)

# Sidebar: Interactive Information Display
with st.sidebar:
    # Sidebar Header with Icon
    st.markdown("<h2 style='color: #4CAF50;'>🧪Chemistry Gestures</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #555;'>Use hand gestures to create compounds and explore their properties.</p>", unsafe_allow_html=True)
    
    # Add a separator for better clarity
    st.markdown("---")

    # Visual Section - Selected Gesture
    st.markdown("<h3 style='color: #4CAF50;'>Current Gesture:</h3>", unsafe_allow_html=True)
    selected_gesture = st.empty()  # Placeholder for gesture detection
    selected_elements_display = st.empty()  # Placeholder for selected elements

    # Display real-time updates for compound info
    st.markdown("<h3 style='color: #4CAF50;'>Current Compound:</h3>", unsafe_allow_html=True)
    compound_info_display = st.empty()  # Placeholder for compound info

    # Interactive Button for Resetting Gesture
    if st.button('Reset Gesture', use_container_width=True):
        selected_elements.clear()
        selected_gesture.text("Gesture: None")
        selected_elements_display.text("Selected Elements: None")
        compound_info_display.text("Compound: None")

# Streamlit Video display
frame_placeholder = st.empty()

# Start Video capture
cap = cv2.VideoCapture(0)

llm_response = None  # Variable to store LLM response

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color for processing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = hands.process(rgb_frame)

    # Draw hand landmarks and detect gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of extended fingers
            extended_fingers = count_extended_fingers(hand_landmarks.landmark)

            # Map extended fingers to corresponding element
            if extended_fingers in GESTURE_MAPPING:
                element = GESTURE_MAPPING[extended_fingers]
                cv2.putText(frame, f"Detected: {element}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                # Add the element to selected_elements list
                if element not in selected_elements:
                    selected_elements.append(element)

                # Update the sidebar with the current gesture and selected elements
                selected_gesture.text(f"Gesture: {GESTURE_MAPPING[extended_fingers]}")
                selected_elements_display.text(f"Selected Elements: {', '.join(selected_elements)}")

            elif extended_fingers == 0:  # Detect pinch gesture (zero extended fingers)
                # Combine selected elements
                compound = combine_elements(selected_elements)
                if compound:
                    cv2.putText(frame, f"Compound: {compound}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    selected_elements = []  # Reset selected elements after combining
                    
                    # Update the sidebar with the compound information
                    compound_info_display.text(f"Compound: {compound}")

                    # Trigger LLM response after forming the compound
                    if compound:
                        prompt = f"Tell me more about the compound {compound}."
                        llm_response = llm.predict(prompt)  # Get response from the LLM

                    # Show compound properties in the main screen
                    st.subheader(f"Compound: {compound}")
                    if compound in COMPOUND_PROPERTIES:
                        properties = COMPOUND_PROPERTIES[compound]
                        for key, value in properties.items():
                            st.write(f"**{key}:** {value}")
                    
                    # Show LLM response below the title
                    if llm_response:
                        st.subheader("LLM Response:")
                        st.write(f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 8px;'>{llm_response}</div>", unsafe_allow_html=True)

                else:
                    cv2.putText(frame, "Invalid combination!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    selected_elements = []  # Reset for invalid attempt

            # Display the selected elements on the screen
            cv2.putText(frame, f"Selected: {', '.join(selected_elements)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the camera feed and display it with a smaller width
    frame_placeholder.image(frame, channels="BGR", use_container_width=True, width=100)  # Adjust width as needed

    # Exit the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

