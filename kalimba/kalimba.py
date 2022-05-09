import cv2
import numpy as np

def wczytaj_plik(plik):
    img = cv2.resize(cv2.imread(plik), (0, 0), fx=0.2, fy=0.2)
    print(img.shape)
    return img

def HoughLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    print("linia: ",[lines[0]])
    parallel= find_parallel(lines)
    for i in range(0, len(lines)):
        #zakomentować if jeśli potrzeba wszystkich linii
        if i in parallel:
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
    return img

def Circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows=gray.shape[0]
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,param1=30, param2=50,minRadius=15, maxRadius=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            center=(i[0],i[1])
            radius=i[2]
            cv2.circle(img, center, 2, (0, 0, 255),3)
            cv2.circle(img, center, radius, (0,0,255),5)
            #print("center: ", center)
            #print("radius: ", radius)
    return img

def find_parallel(lines):
    parallel=[]
    min_poziomo=1.4
    max_poziomo=1.8
    znaleziono_lewy=False
    znaleziono_prawy = False
    for i in range(0,len(lines)):
        for j in range(0,len(lines)):
            if(i==j):continue
            if(abs(lines[i][0][1]-lines[j][0][1])<0.1):
                if lines[i][0][1]>min_poziomo and lines[i][0][1]<max_poziomo:
                    parallel.append(i)
    #print (parallel)
    return parallel

def CornerDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=np.float32(gray)
    dst=cv2.cornerHarris(gray, 2, 3, 0.04)
    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    for i in range(1, len(corners)):
        print(corners[i])
    img[dst>0.01*dst.max()]=[0,0,255]
    dst = cv2.dilate(dst, None)
    img[dst > 0.02 * dst.max()] = [0, 0, 255]
    return img

def ColorSegmentation(img):
    #wykrywanie blaszek
    # rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hsv_img=cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    #
    # lower= (0,0,255)#(22.5,9,32) #0 0 75 (66,65,43)
    # upper=(250,255,30)#(186,195,189)#(110,255,255)
    # mask = cv2.inRange(img, lower, upper)
    # result = cv2.bitwise_and(img, img, mask=mask)
    # print(result)

    #wykrywanie skóry
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)


    imageYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    result = cv2.bitwise_and(img, img, mask=skinRegionYCrCb)
    return result


def display(plik,img):
    cv2.imshow(plik, img)
    cv2.waitKey(0)
    cv2.imwrite("kalim.jpg", img)

def czy_kalimba(radius, linia):

    return True

if __name__=='__main__':
    #pliki bez palca
#    pliki = ["zdjecia/bp1.jpg", "zdjecia/bp2.jpg", "zdjecia/bp3.jpg", "zdjecia/bp4.jpg", "zdjecia/bp5.jpg","zdjecia/bp6.jpg", "zdjecia/bp7.jpg", "zdjecia/bp8.jpg", "zdjecia/bp9.jpg", "zdjecia/bp10.jpg", "zdjecia/bp11.jpg", "zdjecia/bp12.jpg", "zdjecia/bp13.jpg"]

    #pliki bez palca tylko pod kątem
#     pliki = [ "zdjecia/bp2.jpg", "zdjecia/bp3.jpg", "zdjecia/bp4.jpg", "zdjecia/bp5.jpg",
#              "zdjecia/bp7.jpg", "zdjecia/bp8.jpg",
#              "zdjecia/bp11.jpg", "zdjecia/bp12.jpg"]

    #pliki z placem
    # pliki = ["zdjecia/p1.jpg", "zdjecia/p2.jpg", "zdjecia/p3.jpg", "zdjecia/p4.jpg", "zdjecia/p5.jpg",
    #          "zdjecia/p6.jpg", "zdjecia/p7.jpg", "zdjecia/p8.jpg", "zdjecia/p9.jpg", "zdjecia/p10.jpg",
    #          "zdjecia/p11.jpg", "zdjecia/p12.jpg", "zdjecia/p13.jpg", "zdjecia/p14.jpg",
    #          "zdjecia/p15.jpg", "zdjecia/p16.jpg", "zdjecia/p17.jpg", "zdjecia/p18.jpg"]

    #jeden wybrany plik
    pliki=["zdjecia/p2.jpg"]
    for plik in pliki:
        img=wczytaj_plik(plik)

        #img = Circle(img)
        img=HoughLines(img)

        #img=ColorSegmentation(img)
        #img = CornerDetection(img)
        display(plik, img)