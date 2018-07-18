import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

#### could improve by creating a list of user that is acceptable
# for face in 


dong_image = face_recognition.load_image_file('image/dong.jpeg')
dong_face_encoding = face_recognition.face_encodings(dong_image)[0]

rohan_image = face_recognition.load_image_file('image/rohan.jpeg')
rohan_face_encoding = face_recognition.face_encodings(rohan_image)[0]


# Known Faces
known_face_encodings = [
	dong_face_encoding,
	rohan_face_encoding
]
known_face_names = [
	'Dong Hur',
	'Rohan Jha'
]

 # Declare Global Variable
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
scale_image = 4

while True:
	ret, frame = video_capture.read()
	# quarter size image for faster recognition
	small_frame = cv2.resize(frame, (0, 0), fx=1/scale_image, fy=1/scale_image)
	# converts original BGR color to RGB
	rgb_small_frame = small_frame[:, :, ::-1]

	# analyze every other frame
	if process_this_frame:
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		# check if the face matches
		for face_encoding in face_encodings:
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = 'Unknown'

			if True in matches:
				# list of true and false for each known_face list
				first_match_index = matches.index(True)
				name = known_face_names[first_match_index]

			face_names.append(name)

	process_this_frame = not process_this_frame

	# Display Result
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# scale everything back
		top *= scale_image
		right *= scale_image
		bottom *= scale_image
		left *= scale_image

		# draws a box around correct face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# draws a name below the box
		cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom -6), font, 1.0, (255, 255, 255), 1)

	# display image
	cv2.imshow('Video', frame)

	# quit webcame & exit loop (Hit 'q' to quit)
	if cv2.waitKey(1)  & 0xFF == ord('q'):
		break
 # release webcame when done
video_capture.release()
cv2.destroyAllWindows()