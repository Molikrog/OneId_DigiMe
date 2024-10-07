import os

from django.contrib.auth.decorators import login_required
from django.contrib.messages.storage import default_storage
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login
from django.contrib import messages

from django.contrib.auth.models import User

from .forms import SignUpForm, DocumentsForm, IDCheckForm
from .models import UserInfo, Documents, IDCheck

from deepface import DeepFace
import numpy as np
import cv2
from django.core.files.storage import default_storage


# Home view for login
def home(request, token=0):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('compare_face')
        else:
            messages.error(request, 'Username or password is incorrect')
            return redirect('home')

    if token == 0:
        return render(request, 'login.html')


# Register user view
def register_user(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST, request.FILES)

        if form.is_valid():
            user = form.save()
            face_photo = request.FILES.get('image_field')
            if face_photo:
                user_info = UserInfo.objects.create(user=user, image_field=face_photo)

                try:
                    img_path = default_storage.path(user_info.image_field.name)  # Full path to the image
                    face_image = detect_and_crop_face(img_path)
                    resized_image_path = resize_face_image(face_image)

                    face_vector = DeepFace.represent(img_path=resized_image_path, model_name='Facenet')[0]['embedding']
                    user_info.vector = face_vector
                    user_info.save()
                except Exception as e:
                    messages.error(request, f'Error processing image: {e}')
                    user_info.delete()

            messages.success(request, 'You are now registered. Please log in.')
            return redirect('home')

    else:
        form = SignUpForm()

    return render(request, 'register.html', {"form": form})


@login_required(login_url='home')
def compare_face(request):
    if request.method == 'POST':
        user_info = get_object_or_404(UserInfo, user=request.user)

        stored_vector_str = user_info.vector
        match = compare_face_with_camera(stored_vector_str)

        if match:
            messages.success(request, 'Faces match!')
            return redirect('profile')
        else:
            messages.error(request, 'Faces do not match.')

        return redirect('home')

    return render(request, 'compare_face.html')


def compare_vectors(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)


def compare_face_with_camera(stored_vector_str):
    captured_image_path = capture_image_from_camera()
    new_face_vector = DeepFace.represent(img_path=captured_image_path, model_name='Facenet')[0]['embedding']
    cleaned_vector_str = stored_vector_str.strip('[]')
    stored_face_vector = np.array(list(map(float, cleaned_vector_str.split(','))))
    distance = compare_vectors(new_face_vector, stored_face_vector)
    if distance < 5 :
        if distance > 1:
            print("Faces match!")
            return True
    else:
        print("Faces do not match.")
        return False


def get_image_path(user_info):
    try:
        img_path = default_storage.path(user_info.image_field.name)
        return img_path
    except Exception as e:
        print(f"Error retrieving image path: {e}")
        return None


def detect_and_crop_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image. Check the image path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Crop the first face found
    for (x, y, w, h) in faces:
        face_image = image[y:y + h, x:x + w]
        break  # Only use the first detected face

    return face_image


# Function to resize the cropped face image
def resize_face_image(face_image, target_size=(160, 160)):
    resized_face = cv2.resize(face_image, target_size)
    resized_image_path = "resized_face_image.jpg"
    cv2.imwrite(resized_image_path, resized_face)
    return resized_image_path


# Capture image from camera, detect and crop face, then resize
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press Space to Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image_path = 'captured_image.jpg'
            cv2.imwrite(image_path, frame)
            break
    cap.release()
    cv2.destroyAllWindows()

    # Detect face and resize the cropped face
    face_image = detect_and_crop_face(image_path)
    return resize_face_image(face_image)

# Profile view
@login_required(login_url='home')
def profile(request):


    return render(request, 'profile.html')

@login_required(login_url='home')
def logout(request):
    return redirect('home')

@login_required
def documents(request):
    try:
        user_docs = Documents.objects.get(user=request.user)
    except Documents.DoesNotExist:
        user_docs = None

    if request.method == 'POST':
        form = DocumentsForm(request.POST, request.FILES, instance=user_docs)
        if form.is_valid():
            docs = form.save(commit=False)
            docs.user = request.user

            if 'government_id' in request.FILES:
                if user_docs and user_docs.government_id:
                    user_docs.government_id.delete()
                docs.government_id = request.FILES['government_id']

            if 'drivers_license' in request.FILES:
                if user_docs and user_docs.drivers_license:
                    user_docs.drivers_license.delete()
                docs.drivers_license = request.FILES['drivers_license']

            if 'index' in request.FILES:
                if user_docs and user_docs.index:
                    user_docs.index.delete()
                docs.index = request.FILES['index']

            if 'medical_insurance' in request.FILES:
                if user_docs and user_docs.medical_insurance:
                    user_docs.medical_insurance.delete()
                docs.medical_insurance = request.FILES['medical_insurance']

            docs.save()
            return redirect('profile')
    else:
        form = DocumentsForm(instance=user_docs)

    return render(request, 'documents.html', {'form': form, 'user_docs': user_docs})


def id_check_view(request):
    return render(request, 'id_check.html')

def id_check_compare(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        captured_image_path = capture_image_from_camera()  # Captures image and returns the path

        match, user = check_id_face_match(username, captured_image_path)  # Pass image path

        if match:
            messages.success(request, 'Faces match! Access granted to documents.')
            login(request, user)
            return redirect('documents')
        else:
            messages.error(request, 'Faces do not match or user does not exist. Please try again.')

    return render(request, 'id_check.html')

def check_id_face_match(username, captured_image_path):
    try:
        user_info = get_object_or_404(UserInfo, user__username=username)
        stored_vector_str = user_info.vector
        match = compare_face_with_camera_id_check(stored_vector_str, captured_image_path)  # Pass the image path here
        return match, user_info.user
    except UserInfo.DoesNotExist:
        return False, None


def compare_face_with_camera_id_check(stored_vector_str, captured_image_path):
    # Generate the face vector for the newly captured image
    new_face_vector = DeepFace.represent(img_path=captured_image_path, model_name='Facenet')[0]['embedding']

    # Convert the stored vector string into a NumPy array
    cleaned_vector_str = stored_vector_str.strip('[]')
    stored_face_vector = np.array(list(map(float, cleaned_vector_str.split(','))))

    # Compare the two face vectors
    distance = compare_vectors(new_face_vector, stored_face_vector)

    # Check if the distance is within the threshold to consider a match
    if distance < 5:
        if distance > 1:
            print("Faces match!")
            return True
    else:
        print("Faces do not match.")
        return False

