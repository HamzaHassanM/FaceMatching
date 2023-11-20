from deepface import DeepFace
import os

image1 = "kh.jpeg"

models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

all_images_inside_images_folder = "images/"
all_images_inside_images_folder = [os.path.join(all_images_inside_images_folder, image) for image in os.listdir(all_images_inside_images_folder)]

results = []
for image in all_images_inside_images_folder:
    result = DeepFace.verify(image1, image, model_name=models[3], enforce_detection=False)
    if result["distance"] <= 0.5:
        #add image to result 
        result["image"] = image
        results.append(result)

print(results)
