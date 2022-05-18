import cv2
import numpy as np
import glob
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import time
import serial
import cv2.aruco as aruco


def get_map():
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    num1 = 0
    num2 = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        if len(corners) == 4:
            x = (corners[3][0][2][0] - corners[2][0][3][0]) / 42
            y = (corners[3][0][2][1] - corners[2][0][3][1]) / 42

            # cv2.circle(frame_markers, (int(corners[0][0][0][0])-int(x), int(corners[0][0][0][1])-int(y)), 2, 255, -1)
            # cv2.circle(frame_markers, (int(corners[1][0][1][0])+int(x), int(corners[1][0][1][1])-int(y)), 2, 255, -1)
            cv2.circle(frame_markers, (int(corners[2][0][3][0]), int(corners[2][0][3][1])), 2, 255, -1)
            cv2.circle(frame_markers, (int(corners[3][0][2][0]), int(corners[3][0][2][1])), 2, 255, -1)

            # print((corners[3][0][2][0] - corners[2][0][3][0]) / 42)
            # print((corners[3][0][2][1] - corners[2][0][3][1]) / 42)
            point1 = np.float32([[corners[3][0][2][0], corners[3][0][2][1]],
                                 [corners[2][0][3][0], corners[2][0][3][1]],
                                 [corners[1][0][1][0], corners[1][0][1][1]],
                                 [corners[0][0][0][0], corners[0][0][0][1]]])

            point2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
            matrix = cv2.getPerspectiveTransform(point1, point2)

            result = cv2.warpPerspective(frame_markers, matrix, (600, 600))
            cv2.imshow("perspective", result)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                cv2.imwrite("image/picture" + str(num1) + ".jpg", result)
                num1 += 1
                print("get map"+ str(num1))
            elif cv2.waitKey(1) & 0xFF == ord('f'):
                cv2.imwrite("flip_img/picture" + str(num2) + ".jpg", result)
                num2 += 1
                print("get f_img"+ str(num2))
            elif cv2.waitKey(1) & 0xFF == ord('t'):
                cv2.imwrite("card/card.jpg", result)
                print("get tamplate")

        # Display the resulting frame
        cv2.imshow('frame', frame_markers)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
def nothing(x):
    pass

def crop(img):
    size = img.shape
    y = size[0]
    x = size[1]
    cv2.namedWindow('crop')
    cv2.createTrackbar('w_r', 'crop', 0, 1000, nothing)
    cv2.createTrackbar('h_d', 'crop', 0, 1000, nothing)
    cv2.createTrackbar('w_l', 'crop', 0, 1000, nothing)
    cv2.createTrackbar('h_u', 'crop', 0, 1000, nothing)
    while(1):
        wr = cv2.getTrackbarPos('w_r', 'crop')
        hd = cv2.getTrackbarPos('h_d', 'crop')
        wl = cv2.getTrackbarPos('w_l', 'crop')
        hu = cv2.getTrackbarPos('h_u', 'crop')
        crop_img = img[hu:y - hd, wl:x - wr]

        cv2.imshow("crop2", crop_img)

    #Push 'm' to save map.jpg
        if cv2.waitKey(1) & 0xFF == ord('m'):
            cv2.imwrite('map.jpg', crop_img)
            print('crop map finished')
            break

    #Push 't' to save tempate.jpg
        if cv2.waitKey(1) & 0xFF == ord('t'):
            cv2.imwrite('template.jpg', crop_img)
            print('crop template finished')
            break
    cv2.destroyAllWindows()

def preparing_img(num_f_img):
    files = glob.glob("image/*.jpg")
    image_data = []

    for i in range(num_f_img):
        # Flip img
        img = cv2.imread("flip_img/picture"+str(i)+".jpg")
        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("image/picture"+str(i+1000)+".jpg", img_rotate_90_clockwise)
        print("flip img completed")
    for my_file in files:
        this_image = cv2.imread(my_file, 1)
        image_data.append(this_image)

    image_data = np.asarray(image_data)
    dst = np.median(image_data,axis=0).astype(np.uint8)
    print("median finished")
    crop(dst)

def prepaing_template():
    card = cv2.imread('card/card.jpg')
    crop(card)
    # cv2.imshow('card', card)

def edge_detection():
    img = cv2.imread('map.jpg')
    img_b = cv2.imread('map.jpg',0)*0
    # img = cv2.medianBlur(img, 3)

    cv2.namedWindow('config_hsv')
    cv2.createTrackbar('l_h', 'config_hsv', 0, 255, nothing)
    cv2.createTrackbar('l_s', 'config_hsv', 0, 255, nothing)
    cv2.createTrackbar('l_v', 'config_hsv', 0, 255, nothing)
    cv2.createTrackbar('u_h', 'config_hsv', 0, 255, nothing)
    cv2.createTrackbar('u_s', 'config_hsv', 0, 255, nothing)
    cv2.createTrackbar('u_v', 'config_hsv', 0, 255, nothing)
    cv2.namedWindow('image')
    cv2.createTrackbar('canny_H', 'image', 0, 255, nothing)
    cv2.createTrackbar('canny_L', 'image', 0, 255, nothing)
    cv2.createTrackbar('thickness_cnt', 'image', 0, 20, nothing)

    while(1):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lh = cv2.getTrackbarPos('l_h', 'config_hsv')
        ls = cv2.getTrackbarPos('l_s', 'config_hsv')
        lv = cv2.getTrackbarPos('l_v', 'config_hsv')
        uh = cv2.getTrackbarPos('u_h', 'config_hsv')
        us = cv2.getTrackbarPos('u_s', 'config_hsv')
        uv = cv2.getTrackbarPos('u_v', 'config_hsv')

        H = cv2.getTrackbarPos('canny_H', 'image')
        L = cv2.getTrackbarPos('canny_L', 'image')
        t = cv2.getTrackbarPos('thickness_cnt', 'image')

        lower_blue = np.array([lh, ls, lv])
        upper_blue = np.array([uh, us, uv])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        res = cv2.bitwise_and(img,img, mask= mask)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(res,kernel,iterations = 3)
        dilation = cv2.dilate(erosion, kernel, iterations=3)

        cv2.imshow('mask', mask)
        cv2.imshow('hsv', res)
        cv2.imshow('result', dilation)


        gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

        canny = cv2.Canny(gray, H, L)

        cont1, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_c = cv2.drawContours(canny, cont1, -1, 255, t)

        cv2.imshow('canny', img_c)

        cont2, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in cont2:
            cv2.drawContours(canny, [i], -1, 255, -1)
        cv2.imshow('fill', canny)

        skeleton_lee = skeletonize(canny, method='lee')

        cv2.imshow('skel', skeleton_lee)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            cv2.imwrite('path.jpg', dilation)
            cv2.imwrite('skel.jpg', skeleton_lee)
            cv2.imwrite('fill.jpg', canny)
            break
    cv2.destroyWindow()
def polygon_detection_show():
    # img ต้องเป็น gray
    img = cv2.imread('map.jpg', 0)
    img_c = cv2.imread('map.jpg')
    template = cv2.imread('template.jpg', 0)
    black = cv2.imread('map.jpg', 0)*0
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.65
    loc = np.where(res >= threshold)
    for i in zip(*loc[::-1]):
        cv2.rectangle(img_c, i, (i[0] + w, i[1] + h), (0, 0, 255), 1)
        cv2.putText(img_c, "Triangle", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img_c, "({}, {})".format(i[0] + int(w/2), i[1] + int(h/2)), (i[0], i[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(img_c, (i[0] + int(w/2), i[1] + int(h/2)), 1, (0, 0, 255), -1)
        cv2.rectangle(black, i, (i[0] + w, i[1] + h), 255, 1)
    cont, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in cont:
        M = cv2.moments(i)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(black, (cX, cY), 3, 255, -1)

            center_of_symbol = (cX, cY)
    cv2.imshow('polygon', img_c)
    cv2.imshow('polygon_black', black)

            # return center_of_symbol
def polygon_detection():
    # img ต้องเป็น gray
    img = cv2.imread('map.jpg', 0)
    img_c = cv2.imread('map.jpg')
    template = cv2.imread('template.jpg', 0)
    black = cv2.imread('map.jpg', 0)*0
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.65
    loc = np.where(res >= threshold)
    for i in zip(*loc[::-1]):
        cv2.rectangle(img_c, i, (i[0] + w, i[1] + h), (0, 0, 255), 1)
        cv2.putText(img_c, "Triangle", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img_c, "({}, {})".format(i[0] + int(w/2), i[1] + int(h/2)), (i[0], i[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(img_c, (i[0] + int(w/2), i[1] + int(h/2)), 1, (0, 0, 255), -1)
        cv2.rectangle(black, i, (i[0] + w, i[1] + h), 255, 1)
    cont, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in cont:
        M = cv2.moments(i)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(black, (cX, cY), 3, 255, -1)

            center_of_symbol = (cX, cY)
    # cv2.imshow('polygon', img_c)
    # cv2.imshow('polygon_black', black)

            return center_of_symbol
def detect_chess_show():
    img = cv2.imread("map.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 15, 0.01, 10)
    corners = np.int0(corners)

    x, y, w, h = cv2.boundingRect(corners)
    cv2.rectangle(img, (x, y), (x + w, y + h), (23, 0, 255), 3)
    center_of_chess = (int(w/ 2+x), int(h/2+y))
    # cv2.circle(img, (x, y), 3, [0, 255, 0], -1)
    cv2.circle(img, center_of_chess, 3, [0,255,0], -1)
    cv2.imshow("chess", img)
    # return center_of_chess
def detect_chess():
    img = cv2.imread("map.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 15, 0.01, 10)
    corners = np.int0(corners)

    x, y, w, h = cv2.boundingRect(corners)
    cv2.rectangle(img, (x, y), (x + w, y + h), (23, 0, 255), 3)
    center_of_chess = (int(w/ 2+x), int(h/2+y))
    # cv2.circle(img, (x, y), 3, [0, 255, 0], -1)
    cv2.circle(img, center_of_chess, 3, [0,255,0], -1)
    # cv2.imshow("chess", img)
    return center_of_chess

def color_detection_show():
    skel = cv2.imread('skel.jpg',0)
    img = cv2.imread('map.jpg',0)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    img_b = cv2.imread('map.jpg',0)*0
    allpath = cv2.imread('map.jpg', 0) *0
    img_c = cv2.imread('map.jpg')
    img_path = cv2.imread('fill.jpg',0)
    img_c = cv2.fastNlMeansDenoisingColored(img_c, None, 10, 10, 7, 21)

    center_of_symbols = polygon_detection()
    center_of_chess = detect_chess()

    x_list = []
    y_list = []
    x_path = []
    y_path = []
    x_discrete = []
    y_discrete = []
    x_real = []
    y_real = []
    z_list = []
    z_color = []

    for i in range(len(skel)):
        for j in range(len(skel)):
            if skel[i][j] == 255:
                x_list.append(i)
                y_list.append(j)
    x_list = sorted(x_list)
    y_list = sorted(y_list)
    cv2.line(allpath, center_of_symbols, (int(min(y_list)), int(min(x_list))), 255, 1)
    cv2.line(allpath, (int(min(y_list)), int(min(x_list))), (int(max(y_list)), int(max(x_list))), 255, 1)
    cv2.line(allpath, (int(max(y_list)), int(max(x_list))), center_of_chess, 255, 1)

    cv2.imshow('all path', allpath)
    path = cv2.bitwise_and(allpath, img_path)
    cv2.imshow('path', path)


    for i in range(len(path)):
        for j in range(len(path)):
            if path[i][j] == 255:
                x_path.append(i)
                y_path.append(j)
    for i in range(len(x_path)):
        if i % 30 == 0:
            x_discrete.append(x_path[i])
            y_discrete.append(y_path[i])
        else:
            pass
    #########
    #########
    for i in range(len(x_discrete)):
        z1 = img[x_discrete[i],y_discrete[i]]
        z2 = (20-(img[x_discrete[i],y_discrete[i]]*10/255))
        z_color.append(z1)
        z_list.append(z2)
    
    
    for i in range(len(z_color)):
        cv2.circle(img_b, (int(y_discrete[i]), int(x_discrete[i])), 3, (int(z_color[i])), -1)
    
    for i in range(len(x_discrete)):
        cv2.circle(img_c, (int(y_discrete[i]), int(x_discrete[i])), 3, (0, 255, 0), -1)
    
    for i in range(len(z_color)):
        if z_color[i] >= 135:
            z_color[i] = z_color[i+1]
            z_list[i] = z_list[i+1]
    cv2.circle(img_c, center_of_symbols, 3, (0, 0, 255), -1)
    cv2.circle(img_c, center_of_chess, 3, (0, 0, 255), -1)


    z_start_c = z_color[0]

    z_end_c = z_color[-1]
    
    z_start = (20 - (z_start_c * 10 / 255))
    z_end = (20 - (z_end_c * 10 / 255))
    
    cv2.circle(img_b, center_of_symbols, 3, int(z_start_c), -1)
    cv2.circle(img_b, center_of_chess, 3, int(z_end_c), -1)

    # print(img.shape[0])
    # print(z_color)
    # print(z_start_c)
    # print(z_end_c)
    # print(x_discrete)
    # print(y_discrete)


    for i in range(len(x_discrete)):
        x_real.append(int(x_discrete[i]/img.shape[0]*400))
        y_real.append(int(y_discrete[i]/img.shape[1]*400))

    x_start = int(center_of_symbols[1]/img.shape[0]*400)
    y_start = int(center_of_symbols[0]/img.shape[1]*400)
    x_end = int(center_of_chess[1]/img.shape[0]*400)
    y_end = int(center_of_chess[0]/img.shape[1]*400)

    print(int(center_of_symbols[1]/img.shape[0]*400))
    print(int(center_of_symbols[0]/img.shape[1]*400))
    print(x_real)
    print(y_real)
    print(int(center_of_chess[1]/img.shape[0]*400))
    print(int(center_of_chess[0]/img.shape[1]*400))
    cv2.imshow('test_Path', img_c)
    cv2.imshow('intensity', img_b)
    
    
    for i in range(len(z_list)):  # จำนวนตัวในarray
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(x_discrete, y_discrete, z_list)  # เซ็ตพิกัดแกนx,y,zในarray
        plt.show()

def color_detection():
    skel = cv2.imread('skel.jpg',0)
    img = cv2.imread('map.jpg',0)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    img_b = cv2.imread('map.jpg',0)*0
    allpath = cv2.imread('map.jpg', 0) *0
    img_c = cv2.imread('map.jpg')
    img_path = cv2.imread('fill.jpg',0)
    img_c = cv2.fastNlMeansDenoisingColored(img_c, None, 10, 10, 7, 21)

    center_of_symbols = polygon_detection()
    center_of_chess = detect_chess()

    x_list = []
    y_list = []
    x_path = []
    y_path = []
    x_discrete = []
    y_discrete = []
    x_real = []
    y_real = []
    z_list = []
    z_color = []

    for i in range(len(skel)):
        for j in range(len(skel)):
            if skel[i][j] == 255:
                x_list.append(i)
                y_list.append(j)

    x_list = sorted(x_list)
    y_list = sorted(y_list)
    cv2.line(allpath, center_of_symbols, (int(min(y_list)), int(min(x_list))), 255, 1)
    cv2.line(allpath, (int(min(y_list)), int(min(x_list))), (int(max(y_list)), int(max(x_list))), 255, 1)
    cv2.line(allpath, (int(max(y_list)), int(max(x_list))), center_of_chess, 255, 1)

    # cv2.imshow('all path', allpath)
    path = cv2.bitwise_and(allpath, img_path)
    # cv2.imshow('path', path)


    for i in range(len(path)):
        for j in range(len(path)):
            if path[i][j] == 255:
                x_path.append(i)
                y_path.append(j)
    for i in range(len(x_path)):
        if i % 20 == 0:
            x_discrete.append(x_path[i])
            y_discrete.append(y_path[i])
        else:
            pass
    #########
    #########
    for i in range(len(x_discrete)):
        z1 = img[x_discrete[i],y_discrete[i]]
        z2 = int((18-(img[x_discrete[i],y_discrete[i]]*10/200)))
        z_color.append(z1)
        z_list.append(z2)
    
    
    for i in range(len(z_color)):
        cv2.circle(img_b, (int(y_discrete[i]), int(x_discrete[i])), 3, (int(z_color[i])), -1)
    
    for i in range(len(x_discrete)):
        cv2.circle(img_c, (int(y_discrete[i]), int(x_discrete[i])), 3, (0, 255, 0), -1)
    
    # for i in range(len(z_color)):
    if z_color[0] >= 135:
        z_color[0] = z_color[1]
        z_list[0] = z_list[1]
    cv2.circle(img_c, center_of_symbols, 3, (0, 0, 255), -1)
    cv2.circle(img_c, center_of_chess, 3, (0, 0, 255), -1)


    z_start_c = z_color[0]

    z_end_c = z_color[-1]
    
    z_start = int((18 - (z_start_c * 10 / 200)))
    z_end = int((18 - (z_end_c * 10 / 200)))
    
    cv2.circle(img_b, center_of_symbols, 3, int(z_start_c), -1)
    cv2.circle(img_b, center_of_chess, 3, int(z_end_c), -1)

    for i in range(len(x_discrete)):
        x_real.append(int(x_discrete[i]/img.shape[0]*400))
        y_real.append(int(y_discrete[i]/img.shape[1]*400))

    x_start = int(center_of_symbols[1]/img.shape[0]*400)
    y_start = int(center_of_symbols[0]/img.shape[1]*400)
    x_end = int(center_of_chess[1]/img.shape[0]*400)
    y_end = int(center_of_chess[0]/img.shape[1]*400)

    return x_start, y_start, x_real, y_real, x_end, y_end, z_list, z_start, z_end
    
  

# preparing_img()
# prepaing_template()
# print(color_detection())
color_detection_show()
# edge_detection()
# crop(cv2.imread('flip_img/picture13.jpg'))
# finish_point()
# get_map()
# polygon_detection_show()
# detect_chess_show()
# print(polygon_detection())
# print(detect_chess())
# send_data(0x0A, 0x00, 0x00, 0x00, 0x00)
cv2.waitKey(0)

