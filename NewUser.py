import os
import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id = input('Nhập MSSV: ')
print("[INFO] Bắt đầu chụp ảnh, nhìn vào camera!")

sampleNum = 0

while (True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        if cv2.waitKey(1) & 0xFF == 99:
            sampleNum += 1
            cv2.imwrite("dataSet/User." + id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
    text_position = (img.shape[1] - 240, 30)
    cv2.putText(img, f"Images Captured: {sampleNum}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(img, "ESC: exit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "C: take photo", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Capturing Data ', img)

    if cv2.waitKey(50) & 0xFF == 27:
        print("\n[INFO] Đã thoát")
        break
    elif sampleNum >= 20:
        print("\n[INFO] Chụp xong 20 ảnh!")
        break
cam.release()
cv2.destroyAllWindows()
