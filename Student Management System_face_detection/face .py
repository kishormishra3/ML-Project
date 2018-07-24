import face_recognition
import cv2
def call():
    video_capture = cv2.VideoCapture(0)
    kishor_image = face_recognition.load_image_file("F:/k.jpg")
    kishor_face_encoding = face_recognition.face_encodings(kishor_image)[0]
    kishor_face_encoding1 = face_recognition.face_encodings(kishor)[0]
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    name="unknown"
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    
            face_names = []
            for face_encoding in face_encodings:
                match = face_recognition.compare_faces([kishor_face_encoding], face_encoding)
                match1 = face_recognition.compare_faces([kishor_face_encoding1], face_encoding)
                name = "Unknown"
    
                if match[0]:
                    name = "Kishor"
                if match1[0]:
                    name= "Upe"
                face_names.append(name)
    
        process_this_frame = not process_this_frame
    
    
    
        for (top, right, bottom, left), name in zip(face_locations, face_names):
    
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
    
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
    
          
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
        
        cv2.imshow('Video', frame)
    
        if name=="Kishor":
            print("5")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()

call()
