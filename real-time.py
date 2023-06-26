import json
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from deepface import DeepFace
import torch
from torchvision.transforms import transforms
from PIL import Image
from tensorflow.keras.models import model_from_json



# Load the first model: YOLOv7 for face detection
model01 = torch.hub.load('C:\\Users\\amgsoft\\Downloads\\vs_code_yolo\\yolov7', 'custom', 'C:\\Users\\amgsoft\\Downloads\\vs_code_yolo\\yolov7\\best (4).pt', source='local', force_reload=True)

# Load the second model: Anti-spoofing model
#model02 = load_model('yolov7\mobilenetv2-epoch_113.hdf5')


# Load the JSON file

with open('yolov7\\antispoofing_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model architecture from JSON
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights('yolov7\\antispoofing_model.h5')

# Create a file to store the face recognition results
output_file = open('face_recognition_results.txt', 'w')


def preprocess_face(image):
    # Resize the image to the required input shape of the model
    input_shape = (160, 160)
    resized = cv2.resize(image, input_shape)

    # Convert the image to RGB and normalize the pixel values
    normalized = resized / 255.0

    # Add an extra dimension to match the model's input shape
    preprocessed = np.expand_dims(normalized, axis=0)

    return preprocessed

# Define a video capture object
vid = cv2.VideoCapture(0)
count = 0


while True:
    ret, frame = vid.read()
    count+= 1
    if count % 2 == 0:
        
    # Capture the video frame by frame

        # Detect the face in the frame using YOLOv7
        results = model01(frame)

        df = results.pandas().xyxy[0]
        for _, row in df.iterrows():
            if row['class'] == 0 and row['confidence'] > 0.5:
                # Extract the bounding box coordinates
                xmin = int(row['xmin'])
                ymin = int(row['ymin'])
                xmax = int(row['xmax'])
                ymax = int(row['ymax'])

                # Draw a rectangle around the face
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Extract the face region as a separate image
                face_img = frame[ymin:ymax, xmin:xmax]
                preprocessed_face = preprocess_face(face_img)

                # Make a prediction using the loaded anti-spoofing model
                prediction = model.predict(preprocessed_face)
                print(prediction)

                # Classify the prediction as real or spoof based on a threshold
                threshold = 0.5  # Adjust the threshold as per your model and requirements
                if prediction < threshold:
                    label = "Real"
                    #face recognition
                    dfs = DeepFace.find(img_path=frame, enforce_detection=False, db_path="imgs\Camera Roll",
                                        model_name='Facenet512')
                    identities = [result["identity"] for result in dfs]

                    # Print the identities
                    for identity in identities:
                        print(identity)
                        #stock all the identities on dataframe
                    df_identities = pd.DataFrame({"identity": [result["identity"] for result in dfs]})
                    output_file.write(f"Face Recognition Result: {dfs}\n")
                    print(dfs)
                    #print(dfs[1])
                   #Append the result to the current result dataframe
                   
                else:
                    label = "Spoof"
                # Save the overall result dataframe to a file or use it as per your requirement
                df_identities .to_csv('resultrealtime.csv', index=False)
                # Read the CSV file
                df = pd.read_csv('resultrealtime.csv')
                # Display the label on the frame
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
    cv2.imshow('frame', frame)

    # Check for the 'q' button press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
vid.release()

# Close all windows
cv2.destroyAllWindows()

        
# import sqlite3
# import pandas as pd
# from deepface import DeepFace

# # Perform face recognition using DeepFace's find function
# dfs = DeepFace.find(img_path=frame, enforce_detection=False, db_path="imgs\Camera Roll", model_name='Facenet512')

# # Extract the identities from the dfs result
# identities = [result["identity"] for result in dfs]

# # Create a DataFrame to store the identities
# df_identities = pd.DataFrame({"identity": identities})

# # Connect to the SQLite database
# conn = sqlite3.connect("your_database.db")
# cursor = conn.cursor()

# # Insert the identities into the "empattendances" table
# for identity in df_identities["identity"]:
#     query = f"INSERT INTO empattendances (identity) VALUES ('{identity}')"
#     cursor.execute(query)

# # Commit the changes and close the connection
# conn.commit()
# conn.close()

           
           
            
            
            

            
    




