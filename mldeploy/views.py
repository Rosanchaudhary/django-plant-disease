from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.preprocessing import image
import numpy as np

loaded_model = tf.keras.models.load_model('./models/my_model.h5')
 

# Create your views here.
def index(request):
    context = {"a": 1}
    return render(request, "index.html", context)

@csrf_exempt 
def predictImage(request):
    if request.method == 'POST':
        fileObj = request.FILES["file"]
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        testimage = "." + filePathName
        print(testimage)
        preprocessed_image = preprocess_image(testimage)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(preprocessed_image)

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)

        # List of class labels corresponding to your dataset
        class_labels = ['BrownSpot', 'Healthy', 'Hispa', 'Hispa']

        # Get the predicted label based on the index
        predicted_label = class_labels[predicted_class_index]

        context = {
            "filePathName": filePathName,
            "predictedLabel": predicted_label,
        }
        return render(request, "index.html", context) 


# Define a function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array