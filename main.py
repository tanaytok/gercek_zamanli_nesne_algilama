import cv2

nesne = cv2.imread("yaprak.jpg")
nesne = cv2.cvtColor(nesne, cv2.COLOR_BGR2GRAY)
nesne = cv2.resize(nesne, (400, 299))

kamera = cv2.VideoCapture(0)

while True:
    ret, kare = kamera.read()
    if not ret:
        break

    gri_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(gri_kare, nesne, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + 400, top_left[1] + 299)

    cv2.rectangle(kare, top_left, bottom_right, (0, 0, 255), 2)

    cv2.imshow("Yaprak Tanıma", kare)

    if cv2.waitKey(1) == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()