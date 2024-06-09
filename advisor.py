import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image
import os
import pandas as pd

# Load frame suggestions
frame_suggestions = pd.read_csv('frame_suggestions.csv')

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# Function to detect landmarks
def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return landmarks


# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to predict face shape
def predict_face_shape(landmarks):
    if landmarks is None:
        return None

    jaw_points = [landmarks.part(i) for i in range(0, 17)]
    jaw_width = euclidean_distance((jaw_points[0].x, jaw_points[0].y), (jaw_points[16].x, jaw_points[16].y))
    cheek_width = euclidean_distance((landmarks.part(1).x, landmarks.part(1).y),
                                     (landmarks.part(15).x, landmarks.part(15).y))
    face_height = euclidean_distance((landmarks.part(8).x, landmarks.part(8).y),
                                     (landmarks.part(27).x, landmarks.part(27).y))
    forehead_width = euclidean_distance((landmarks.part(19).x, landmarks.part(19).y),
                                        (landmarks.part(24).x, landmarks.part(24).y))
    cheekbone_width = euclidean_distance((landmarks.part(3).x, landmarks.part(3).y),
                                         (landmarks.part(13).x, landmarks.part(13).y))

    # Additional landmarks
    left_eye_width = euclidean_distance((landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(39).x, landmarks.part(39).y))
    right_eye_width = euclidean_distance((landmarks.part(42).x, landmarks.part(42).y),
                                         (landmarks.part(45).x, landmarks.part(45).y))
    nose_to_mouth_distance = euclidean_distance((landmarks.part(30).x, landmarks.part(30).y),
                                                (landmarks.part(33).x, landmarks.part(33).y))
    mouth_width = euclidean_distance((landmarks.part(48).x, landmarks.part(48).y),
                                     (landmarks.part(54).x, landmarks.part(54).y))

    # Calculate ratios
    width_to_height_ratio = jaw_width / face_height
    cheek_to_jaw_ratio = cheek_width / jaw_width
    cheekbone_to_jaw_ratio = cheekbone_width / jaw_width
    forehead_to_jaw_ratio = forehead_width / jaw_width
    eye_width_ratio = (left_eye_width + right_eye_width) / (2 * jaw_width)
    nose_to_mouth_ratio = nose_to_mouth_distance / face_height
    mouth_width_ratio = mouth_width / jaw_width

    # New improved logic
    if cheek_to_jaw_ratio > 1.1 and cheekbone_to_jaw_ratio > 1.05 and forehead_to_jaw_ratio > 1.1:
        return "Heart"
    elif width_to_height_ratio < 0.9 and cheek_to_jaw_ratio < 1.0 and cheekbone_to_jaw_ratio < 1.05:
        return "Rectangle"
    elif width_to_height_ratio > 1.0 and cheek_to_jaw_ratio > 1.1 and cheekbone_to_jaw_ratio > 1.05:
        return "Round"
    elif forehead_to_jaw_ratio < 0.8 and cheekbone_to_jaw_ratio > 1.05:
        return "Diamond"
    else:
        return "Oval"


# Function to suggest frames based on face shape, age, and gender
def suggest_frames(face_shape, age, gender):
    suggestions = frame_suggestions[
        (frame_suggestions['face_shape'] == face_shape) &
        (frame_suggestions['min_age'] <= age) &
        (frame_suggestions['max_age'] >= age) &
        (frame_suggestions['gender'] == gender)
        ]
    return suggestions['frame_image'].tolist()


# Function to overlay glasses
def overlay_glasses(image, landmarks, glasses):
    left_eye_center = np.mean(
        [(landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(39).x, landmarks.part(39).y)], axis=0).astype(
        int)
    right_eye_center = np.mean(
        [(landmarks.part(42).x, landmarks.part(42).y), (landmarks.part(45).x, landmarks.part(45).y)], axis=0).astype(
        int)

    eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype(int)

    eye_width = np.linalg.norm(np.array(right_eye_center) - np.array(left_eye_center))
    face_width = np.abs(left_eye_center[0] - right_eye_center[0]) * 2
    scale_factor = face_width / (glasses.shape[1] * 0.85)  # Scale factor to fit the glasses based on face width
    new_glasses_size = (int(glasses.shape[1] * scale_factor), int(glasses.shape[0] * scale_factor))

    resized_glasses = cv2.resize(glasses, new_glasses_size, interpolation=cv2.INTER_AREA)

    x_offset = eye_center[0] - int(resized_glasses.shape[1] / 2)
    y_offset = eye_center[1] - int(resized_glasses.shape[0] / 2)

    y1, y2 = y_offset, y_offset + resized_glasses.shape[0]
    x1, x2 = x_offset, x_offset + resized_glasses.shape[1]

    # Ensure the overlay stays within image boundaries
    y1 = max(y1, 0)
    y2 = min(y2, image.shape[0])
    x1 = max(x1, 0)
    x2 = min(x2, image.shape[1])

    alpha_glasses = resized_glasses[:, :, 3] / 255.0
    alpha_face = 1.0 - alpha_glasses

    for c in range(0, 3):
        image[y1:y2, x1:x2, c] = (alpha_glasses * resized_glasses[:, :, c] +
                                  alpha_face * image[y1:y2, x1:x2, c])


# Function to display images in a grid
def display_images_in_grid(images, captions, columns=3):
    rows = len(images) // columns + (1 if len(images) % columns else 0)
    for row in range(rows):
        cols = st.columns(columns)
        for col in range(columns):
            idx = row * columns + col
            if idx < len(images):
                with cols[col]:
                    st.image(images[idx], caption=captions[idx], use_column_width=True)


# Streamlit UI
st.title("Frameify")

# CSS for styling
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding: 1rem 1rem 1rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User inputs
name = st.text_input("Enter your name")
gender = st.selectbox("Enter your gender", ["Male", "Female"])
age = st.number_input("Enter your age", min_value=0, max_value=100, value=25)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    user_image = Image.open(uploaded_file)
    user_image = np.array(user_image)

    landmarks = detect_landmarks(user_image)
    if landmarks is None:
        st.write("No face detected.")
    else:
        face_shape = predict_face_shape(landmarks)
        if face_shape:
            st.write(f"Detected face shape: {face_shape}")
            suggested_frames = suggest_frames(face_shape, age, gender)
            if suggested_frames:
                st.write("Trying suggested frames...")
                frames_folder = "frames"
                images = []
                captions = []
                for frame_file in suggested_frames:
                    frame_path = os.path.join(frames_folder, frame_file)
                    glasses_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                    result_image = user_image.copy()
                    overlay_glasses(result_image, landmarks, glasses_image)
                    images.append(result_image)
                    captions.append(f"With {frame_file}")
                display_images_in_grid(images, captions)
            else:
                st.write("No frame suggestions available for your profile.")
        else:
            st.write("Could not determine face shape.")

# Camera capture
camera_image = st.camera_input("Take a picture")
if camera_image is not None:
    user_image = Image.open(camera_image)
    user_image = np.array(user_image)

    landmarks = detect_landmarks(user_image)
    if landmarks is None:
        st.write("No face detected.")
    else:
        face_shape = predict_face_shape(landmarks)
        if face_shape:
            st.write(f"Detected face shape: {face_shape}")
            suggested_frames = suggest_frames(face_shape, age, gender)
            if suggested_frames:
                st.write("Trying suggested frames...")
                frames_folder = "frames"
                images = []
                captions = []
                for frame_file in suggested_frames:
                    frame_path = os.path.join(frames_folder, frame_file)
                    glasses_image = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                    result_image = user_image.copy()
                    overlay_glasses(result_image, landmarks, glasses_image)
                    images.append(result_image)
                    captions.append(f"With {frame_file}")
                display_images_in_grid(images, captions)
            else:
                st.write("No frame suggestions available for your profile.")
        else:
            st.write("Could not determine face shape.")
