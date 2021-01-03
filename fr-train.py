from PIL import Image
import os,numpy as np, cv2

path = 'knownFaces'

face_recognizer= cv2.face.LBPHFaceRecognizer_create()
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def Label_Images(path):
	image_paths=[os.path.join(path,f) for f in os.listdir(path)]
	samples=[]
	ids=[]
	for image_path in image_paths:
		image_grey=Image.open(image_path).convert('L')
		image_numpy= np.array(image_grey,'uint8')
		id= int(os.path.split(image_path)[-1].split(".")[1])
		faces=face_cascade.detectMultiScale(image_numpy)
		for x,y,w,h in faces:
			samples.append(image_numpy[y:y+h,x:x+w])
			ids.append(id)
	return samples,ids

samples,ids=Label_Images(path)
label_id=np.array(ids)
face_recognizer.train(samples,label_id)
face_recognizer.write('trained.yml')

