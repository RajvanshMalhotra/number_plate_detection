import cv2

count = 0
cap = cv2.VideoCapture("demo.mp4")
number_plate_classifier = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    if ret:
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        number_plates = number_plate_classifier.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in number_plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame, "number plate", (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
            frame_roi = frame[y:y + h, x:x + w]  

        cv2.imshow("frame ROI", frame_roi)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if frame_roi is not None:
                cv2.imwrite(f"pics{count}.jpg", frame_roi)
                cv2.rectangle(frame, (0, 100), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "snapped!", (150, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Output Video", frame)
                cv2.waitKey(1000) 
                count += 1
        elif key == ord('e'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
