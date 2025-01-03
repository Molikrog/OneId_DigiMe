# OneId_DigiMe
Introduction 

The **OneID Digital Identity Management System** is a user authentication system based on biometrics. Users utilize the application to securely store and input confidential data (e.g., ID cards, driver’s licenses, student indexes, etc.) and gain secure access to them. The application allows registration through a username and password, followed by a facial authentication process using facial recognition technology. Through this precise verification, users can securely access their confidential data. Additionally, the OneID functionality enables authentication using only a username and a photo. 

Key Features: 
    
	• Biometric Authentication: Uses facial recognition technology for user authentication.
    
	• Intuitive Interface: A simple and user-friendly interface for easy access to documents.
    
	• Camera-Based Authentication: Secure authentication using the device’s camera.
    
	• Secure Document Storage: Grants access to user documents upon successful verification.
    
	• Django Framework: The project is built using Django. 

Technologies Used: 

	• Django: Framework used to develop the project.
    
	• DeepFace: Library for facial recognition and emotion detection.
    
	• OpenCV: Library for image processing and manipulation.
    
	• SQLite: Database used for storing and verifying user data. 

 # Code Overview and Infrastructure

 The infrastructure of the application is shown below, where the **"app"** application contains all the main components of the program and its functionality. The main project, named **"OneID_Django"**, includes the essential configurations: "settings.py" and **"urls.py"**, where **"settings.py"** contains the main file for all important project configurations and **"urls.py"** contains the mapping of all URLs in the project, linking them to the corresponding **"views"**. Additionally, we have the database where all the data is stored. 

**The project structure is as follows**:
    
	 • Main project directory: OneID_Django
    
	 • Contains essential configurations (settings.py and urls.py).
        	
		 ◦ settings.py: Stores primary configurations.
        	
		 ◦ urls.py: Maps URLs to corresponding views.
 	
  		
	• Application directory: app
        	
		 ◦ Contains all main components and functionalities.
		
  	• Database: Stores user data securely. 

# Registration Process

When a user starts the program, they must first register. The registration process ensures that users are stored in the database before they can use the application. The system uses Django forms for structured data input.
	
 Forms: Used for validating and processing user input via HTML forms.
	
 Form-data: Data format used for sending user input to the server via HTTP POST.

During registration, the system enforces certain criteria for user input. The Django authentication system simplifies registration, while additional functions handle image verification by detecting, cropping, resizing, and vectorizing the user’s face.

	 def register_user(request):
	    if request.method == 'POST':
	        form = SignUpForm(request.POST, request.FILES)
	        if form.is_valid():
	            user = form.save()
	            face_photo = request.FILES.get('image_field')
	            if face_photo:
	                user_info = UserInfo.objects.create(user=user, image_field=face_photo)
	
	                try:
	                    img_path = default_storage.path(user_info.image_field.name)
	                    face_image = detect_and_crop_face(img_path)
	                    resized_image_path = resize_face_image(face_image)
	                    face_vector = DeepFace.represent(img_path, model_name="Facenet")[0]['embedding']
	                    user_info.vector = face_vector
	                    user_info.save()
	                except Exception as e:
	                    messages.error(request, f'Error processing image: {e}')
	                    user_info.delete()
	
	            messages.success(request, "You are now registered. Please log in.")
	            return redirect('home')
	    else:
	        form = SignUpForm()
	    return render(request, 'register.html', {'form': form})

  Upon successful registration, users receive a confirmation message and can proceed to log in.

# Login Process

  The login function processes user credentials and redirects them based on authentication success or failure.

			def home(request, token=8):
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
	
	    return render(request, 'login.html')

  Upon successful login, users are redirected to the biometric verification page.

# Biometric Verification

  The core of this project is face authentication, ensuring the most secure user verification method. The process involves:

	1. Capturing a new image from the user's camera.
	2. Extracting the user’s unique face vector.
	3. Comparing it against the stored vector in the database.
	4. Granting or denying access based on the similarity score.

	@login_required(login_url="home")
	def compare_face(request):
	    if request.method == 'POST':
	        user_info = get_object_or_404(UserInfo, user=request.user)
	        stored_vector_str = user_info.vector
	        match = compare_face_with_camera(stored_vector_str)
	
	        if match:
	            messages.success(request, "Faces match!")
	            return redirect('profile')
	        else:
	            messages.error(request, "Faces do not match.")
	            return redirect('home')
	
	    return render(request, 'compare_face.html')

  The function compare_face_with_camera captures an image from the camera, processes it, and checks if it matches the stored facial vector.

	  def compare_face_with_camera(stored_vector_str):
	    captured_image_path = capture_image_from_camera()
	    new_face_vector = DeepFace.represent(img_path=captured_image_path, model_name="Facenet")[0]['embedding']
	    stored_face_vector = np.array(list(map(float, stored_vector_str.strip('[]').split(','))))
	    distance = np.linalg.norm(new_face_vector - stored_face_vector)
	
	    return distance < 5

  If the distance between vectors is within a valid range, access is granted.

# Document Management
After successful authentication, users can store and manage documents. The document management system allows uploading images of various documents (ID, driver's license, etc.).
	
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
	            docs.save()
	            return redirect('profile')
	
	    else:
	        form = DocumentsForm(instance=user_docs)
	
	    return render(request, 'documents.html', {'form': form, 'user_docs': user_docs})

# OneID Authentication Without Password

The system also allows authentication using only a username and face recognition, without a password. The process retrieves the user's face vector based on their username and performs verification.

	def id_check_compare(request):
	    if request.method == 'POST':
	        username = request.POST.get('username')
	        captured_image_path = capture_image_from_camera()
	        match, user = check_id_face_match(username, captured_image_path)
	
	        if match:
	            messages.success(request, "Faces match! Access granted to documents.")
	            login(request, user)
	            return redirect('documents')
	        else:
	            messages.error(request, "Faces do not match or user does not exist.")
	            return render(request, 'id_check.html')

# Conclusion

The **OneID Digital Identity Management System** provides a secure and flexible way for users to store and manage their personal documents while ensuring biometric authentication. Instead of carrying physical documents, users can access and present them securely using facial recognition technology.

This system enhances both security and user convenience, making authentication fast, reliable, and efficient.





