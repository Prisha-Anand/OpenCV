import cv2
import numpy as np
import face_detect as face_detect
import training_data as training_data

label = []
def predict(test_img):
	img = cv2.imread(test_img).copy()
	print("\n\n\n")
	print("Face Prediction Running -\-")
	face, rect, length = face_detect.face_detect(test_img)
	print (len(face), "faces detected.")
	for i in range(0, len(face)):
		labeltemp, confidence = face_recognizer.predict(face[i])
		label.append(labeltemp)
	return img, label

faces, labels = training_data.training_data("\Pictures\faces")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


# Read the test image.
img2 = cv.VideoCapture(0)
(_, img)=img2.read()
predicted_img , label= predict(img)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
print ("Recognized faces = ", label)
