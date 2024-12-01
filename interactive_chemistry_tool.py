# import cv2
# import numpy as np
# import pyttsx3
# import tkinter as tk
# from PIL import Image, ImageTk
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv

# # Load environment variables for API keys
# load_dotenv()

# # Initialize Text-to-Speech Engin
# tts_engine = pyttsx3.init()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# # Gesture-to-Element Mapping
# gesture_to_element = {
#     "one_finger": "Oxygen (O)",
#     "two_fingers": "Hydrogen (H)"
# }

# # Selected Elements
# selected_elements = []

# # Function for Text-to-Speech
# def narrate(text):
#     tts_engine.say(text)
#     tts_engine.runAndWait()

# # Function to Count Fingers Using Convexity Defects
# def count_fingers(contour):
#     hull = cv2.convexHull(contour, returnPoints=False)
#     defects = cv2.convexityDefects(contour, hull)

#     if defects is None:
#         return 0

#     finger_count = 0
#     for i in range(defects.shape[0]):
#         start, end, far, _ = defects[i, 0]
#         start = tuple(contour[start][0])
#         end = tuple(contour[end][0])
#         far = tuple(contour[far][0])

#         # Distance calculation
#         a = np.linalg.norm(np.array(start) - np.array(far))
#         b = np.linalg.norm(np.array(end) - np.array(far))
#         c = np.linalg.norm(np.array(start) - np.array(end))
#         angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

#         # If the angle is less than 90 degrees, it's a valid finger
#         if angle <= np.pi / 2:
#             finger_count += 1

#     return finger_count + 1  # Add thumb

# def detect_gesture(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_skin = np.array([0, 30, 60], dtype=np.uint8)
#     upper_skin = np.array([20, 150, 255], dtype=np.uint8)
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)

#     # Morphological transformations
#     mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
#     mask = cv2.GaussianBlur(mask, (5, 5), 100)

#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         max_contour = max(contours, key=cv2.contourArea)
#         if cv2.contourArea(max_contour) > 3000:  # Reduced minimum area
#             return count_fingers(max_contour)

#     return None

# # LangChain Setup
# prompt_template = """
# Provide a detailed explanation about the following chemical element or compound: {text}.
# Include its uses, properties, and significance.
# """
# prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
# llm_chain = prompt | llm

# # Function to Generate Explanations Using LangChain
# def generate_explanation(text):
#     try:
#         explanation = llm_chain.invoke({"text": text})
#         return explanation
#     except Exception as e:
#         return f"Error generating explanation: {e}"

# # Function to Update Tkinter GUI
# def update_gui(elements, explanation=None):
#     for widget in frame.winfo_children():
#         widget.destroy()

#     label = tk.Label(frame, text="Selected Elements:", font=("Arial", 14))
#     label.pack()

#     for element in elements:
#         element_label = tk.Label(frame, text=f"- {element}", font=("Arial", 12))
#         element_label.pack()

#     if explanation:
#         explanation_label = tk.Label(frame, text=f"Explanation:\n{explanation}", font=("Arial", 12), fg="blue", wraplength=400, justify="left")
#         explanation_label.pack()

# # Create Tkinter Window
# window = tk.Tk()
# window.title("Interactive Chemistry Tool")
# window.geometry("600x700")

# frame = tk.Frame(window)
# frame.pack(pady=20)

# label_img = tk.Label(window)
# label_img.pack()

# cap = cv2.VideoCapture(0)

# while True:
#     success, frame_cv = cap.read()
#     if not success:
#         print("Failed to capture frame.")
#         break

#     frame_cv = cv2.flip(frame_cv, 1)
#     gesture = detect_gesture(frame_cv)

#     if gesture in gesture_to_element:
#         element = gesture_to_element[gesture]
#         if element not in selected_elements:
#             selected_elements.append(element)
#             narrate(f"{element} selected")

#     compound = None
#     if "Oxygen (O)" in selected_elements and "Hydrogen (H)" in selected_elements:
#         compound = "H2O"
#         selected_elements.clear()

#     explanation = None
#     if compound:
#         explanation = generate_explanation(compound)
#     elif selected_elements:
#         explanation = generate_explanation(" and ".join(selected_elements))

#     update_gui(selected_elements, explanation)

#     img = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
#     img_pil = Image.fromarray(img)
#     img_tk = ImageTk.PhotoImage(img_pil)

#     label_img.config(image=img_tk)
#     label_img.image = img_tk

#     window.update_idletasks()
#     window.update()

# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp

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
    'pinch': 'Combine Elements',  # Pinch gesture for combining elements
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
                cv2.putText(frame, f"Detected: {element}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Add the element to selected_elements list
                if element not in selected_elements:
                    selected_elements.append(element)
                    cv2.putText(frame, f"Selected: {element}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif extended_fingers == 0:  # Detect pinch gesture (zero extended fingers)
                # Combine selected elements
                compound = combine_elements(selected_elements)
                if compound:
                    cv2.putText(frame, f"Compound: {compound}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    selected_elements = []  # Reset selected elements after combining
                else:
                    cv2.putText(frame, "Invalid combination!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the video feed
    cv2.imshow("Interactive Chemistry Tool", frame)

    # Exit the tool with the 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
