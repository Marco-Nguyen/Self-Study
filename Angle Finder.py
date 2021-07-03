import cv2
import math
# hoang dinh? qua'
# 'hello'
# hello helolo thing gioi qua
path = 'degrees.png'

img = cv2.imread(path)
pointsList = []
cap = cv2.VideoCapture(0)
myColor = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(img, tuple(
                pointsList[round((size - 1) / 3) * 3]), (x, y), myColor[3], 2)
        cv2.circle(img, (x, y), 5, myColor[2], -1)
        # print(x, y)
        pointsList.append([x, y])
        print(pointsList)

# def gradient(pt1, pt2):
#     return (pt2[1] - pt1[1])/(pt2[0] - pt1[0])


def vector_length(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)


# def getAngle(pointsList):
#     # print("angle")
#     pt1, pt2, pt3 = pointsList[-3:]
#     print(pt1, pt2, pt3)
#     m1 = gradient(pt1, pt2)
#     m2 = gradient(pt1, pt3)
#     print(m1, m2)
#     angleR = math.atan((m2 - m1)/(1 + m1*m2))
#     angleD = round(math.degrees(angleR))
#     cv2.putText(img, str(angleD), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_ITALIC, 1.5, (0, 0, 255), 2)


def getAngle(pointsList, image):
    # print("angle")
    pt1, pt2, pt3 = pointsList[-3:]
    # print(pt1, pt2, pt3)
    # m1 = (pt2[0] - pt1[0])*(pt3[0] - pt1[0])
    # m2 = (pt2[1] - pt1[1])*(pt3[1] - pt1[1])
    m1 = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
    m2 = [pt3[0] - pt1[0], pt3[1] - pt1[1]]
    # print(m1, m2)
    angleR = math.acos((m1[0] * m2[0] + m1[1] * m2[1]) /
                       (vector_length(m1) * vector_length(m2)))
    angleD = round(math.degrees(angleR))
    cv2.putText(image, str(angleD),
                (pt1[0] - 40, pt1[1] - 20), cv2.FONT_ITALIC, 1.5, (0, 0, 255), 2)


while True:
    success, frame = cap.read()
    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList, img)

    cv2.imshow('Frame', img)
    cv2.setMouseCallback('Frame', mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        pointsList = []
        img = cv2.imread(path)
        count = 0

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
