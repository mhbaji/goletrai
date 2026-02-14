from goletrai import GoletRai
import cv2 

if __name__ == '__main__':
    grai = GoletRai()

    vPlayer = cv2.VideoCapture(0)
    if not vPlayer.isOpened():
        exit()

    while True:
        ret, image = vPlayer.read()
        if not ret: break

        results = grai.ning(image)
        image = grai.gambar(image, results)

        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == ord('q'): break