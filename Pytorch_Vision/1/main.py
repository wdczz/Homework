import cv2


def draw(cnt,img,count):
    x,y,w,h=cv2.boundingRect(cnt)
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    img=cv2.putText(img,str(count),(x,y),cv2.FONT_HERSHEY_PLAIN,0.5,(0,255,0))
    return img


if __name__ == '__main__':
    cv2.namedWindow("show", cv2.WINDOW_GUI_NORMAL)
    img=cv2.imread("img.png",1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    cnts,_=cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for index,cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > 20:
            img=draw(cnt,img,index)
    cv2.imshow("show",img)
    cv2.waitKey()