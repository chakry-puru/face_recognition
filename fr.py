import cv2,numpy as np, os

face_recognizer= cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained.yml')
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id=0
face_names=['None','Pururav','x','y']
cap=cv2.VideoCapture(0)

while(True):
	_,frame=cap.read()
	frame_grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces= face_cascade.detectMultiScale(frame_grey)
	for x,y,w,h in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		id,confidence= face_recognizer.predict(frame_grey[y:y+h,x:x+w])
		if confidence<100:
			id=face_names[id]
		else:
			id='unknown'
		cv2.putText(frame,str(id),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
		cv2.imshow('frame',frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
cap.release()
cv2.destroyAllWindows()
			
