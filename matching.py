from deepface import DeepFace
import numpy as np
# Load an image
image_path = "../1.jpg"

# Extract face embeddings
embedding1 = DeepFace.represent(img_path=image_path) ## returns a list of 1 dictionary


embedding2 = DeepFace.represent(img_path="../2.jpeg") ## returns a list of 1 dictionary

embedding1_np = np.array(embedding1[0]["embedding"]) ## 1x128 vector
embedding2_np = np.array(embedding2[0]["embedding"]) ## 1x128 vector


# Calculate the Euclidean distance between the two embeddings (L2 norm) formula: sqrt(sum(square(x-y))) 
euclidean_distance = np.linalg.norm(embedding1_np - embedding2_np) 

# Set a threshold for similarity (you can adjust this based on your needs)
similarity_threshold = 0.6 

# Check if the faces are similar based on the threshold
if euclidean_distance < similarity_threshold:
    print("Faces are similar.")
    print("Euclidean distance:", euclidean_distance)
else:
    print("Faces are not similar.")
