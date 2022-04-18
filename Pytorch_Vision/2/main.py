import cv2
from rich.console import Console

def draw(cnt,img,count):
    x,y,w,h=cnt
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    img=cv2.putText(img,str(count),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(0,255,0))
    return img

def callback(object):
    pass

if __name__ == '__main__':
    console=Console()
    cv2.namedWindow("show", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("track", cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar('scaleFactor', "track", 101, 255, callback)
    cv2.createTrackbar('minNeighbors', "track", 1, 10, callback)
    img=cv2.imread("img.png",cv2.COLOR_BGR2RGB)
    haar = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
    while True:
        img_copy=img.copy()
        scaleFactor=cv2.getTrackbarPos('scaleFactor', 'track')
        if scaleFactor>100:
            scaleFactor=scaleFactor/100
        else:
            scaleFactor=1.2
        minNeighbor = cv2.getTrackbarPos('minNeighbors', 'track')
        img_copy = cv2.putText(img_copy, str(scaleFactor), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        img_copy = cv2.putText(img_copy, str(minNeighbor), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        rects = haar.detectMultiScale(img_copy, scaleFactor=scaleFactor, minNeighbors=minNeighbor)
        for index,cnt in enumerate(rects):
            img_copy=draw(cnt,img_copy,index)
        cv2.imshow("show",img_copy)
        if cv2.waitKey(1)==ord("q"):
            console.log("Quit!")
            break
    cv2.destroyAllWindows()