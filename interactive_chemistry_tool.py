from compound_db import COMPOUND_PROPERTIES
import cv2
import mediapipe as mp
import time


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
    else:
        return None  # Invalid combination

# Function to display compound properties
def display_compound_properties(frame, compound, x, y):
    """Display properties of a compound on the screen."""
    if compound in COMPOUND_PROPERTIES:
        properties = COMPOUND_PROPERTIES[compound]
        for i, (key, value) in enumerate(properties.items()):
            cv2.putText(frame, f"{key}: {value}", (x, y + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Initialize OpenCV for real-time video
cap = cv2.VideoCapture(0)

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

            elif extended_fingers == 0:  # Detect pinch gesture (zero extended fingers)
                # Combine selected elements
                compound = combine_elements(selected_elements)
                if compound:
                    cv2.putText(frame, f"Compound: {compound}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    display_compound_properties(frame, compound, 10, 200)  # Display properties below the compound
                    selected_elements = []  # Reset selected elements after combining
                else:
                    cv2.putText(frame, "Invalid combination!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    selected_elements = []  # Reset for invalid attempt

            # Display the selected elements on the screen
            cv2.putText(frame, f"Selected: {', '.join(selected_elements)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add a background to make it more attractive
    frame = cv2.putText(frame, "Interactive Chemistry Tool", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (5, 50), (frame.shape[1]-5, frame.shape[0]-5), (255, 255, 255), 5)

    # Display the video feed
    cv2.imshow("Interactive Chemistry Tool", frame)

    # Exit the tool with the 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
