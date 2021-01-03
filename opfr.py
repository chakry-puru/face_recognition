import numpy as np, cv2, os

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceid = int(input('enter user id end press <return> ==>  '))
print(faceid)
i=0
cap = cv2.VideoCapture(0)
while True:
	ret, frame= cap.read()
	fgray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(fgray,1.5,5) #scalefactor an min no of neighbours
		
	for x,y,w,h in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		#save the face in an image
		roi_gray= fgray[y:y+h,x:x+w]
		
		#img_item='zzz'+str(i)+'.png' #temp var for empty image
		cv2.imwrite('knownFaces/Pururav.'+str(faceid)+'.'+str(i)+'.png',roi_gray)
		i=i+1 
	cv2.imshow('frame', frame)
	#cv2.imshow('roi_gray',img_item)
	
	k=cv2.waitKey(100)& 0xFF
	if k==27:
		break
	elif i>=30:
		break
	#if cv2.waitKey(1) & 0xFF == ord('q'):
	#	break
cap.release()
cv2.destroyAllWindows()