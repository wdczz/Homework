from rich.console import Console
from predict import *
import time


def draw(cnt,img):
    x,y,w,h=cv2.boundingRect(cnt)
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    return img

def get_patch(img,cnt,dst,index,net_choose):
    x, y, w, h = cv2.boundingRect(cnt)
    output=dst[y-10:y+h+10,x-10:x+w+10]
    result = pre(console, output,index,net_choose)
    img = cv2.putText(img, str(result), (x, y), cv2.FONT_HERSHEY_PLAIN,3, (0, 255, 0))
    return img

if __name__ == '__main__':
    console = Console()
    console.rule("Start")
    while True:
        net_choose = console.input("Choose your net to train  --- 1. Convnet / 2. Lenet / 3. Quit -> ")
        if net_choose=='3':
            break
        elif net_choose!='1' and net_choose!='2' and net_choose!='3':
            console.log("You input a error number {}, so you can try to input 1 \ 2 \ 3!".format(net_choose))
            continue
        while True:
            path = console.input("Please give me the path of the picture -> ")
            img = cv2.imread(path, 1)
            try:
                if img.any() != None:
                    break
            except:
                console.log("Oh the path maybe is not effective ,so give me another one !")
        cv2.namedWindow("show", cv2.WINDOW_GUI_NORMAL)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
        dst = cv2.bitwise_not(dst)
        cnts,_=cv2.findContours(dst, cv2.RETR_EXTERNAL  , cv2.CHAIN_APPROX_SIMPLE)
        print(len(cnts))
        img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
        start=time.time()
        for index,cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            if area > 20:
                img=draw(cnt,img)
                img=get_patch(img,cnt,dst,index,net_choose)
        console.print("Use {}s to predict!".format(time.time()-start))
        cv2.imshow("show",img)
        cv2.waitKey()
    console.rule("End")