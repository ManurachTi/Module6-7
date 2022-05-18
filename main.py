from img_procressing import *
import serial

ser = serial.Serial('com21', 115200)
ser.rts = 0
time.sleep(1)

ar = serial.Serial('com22', 19200,timeout=1)
time.sleep(3)
print(ar.isOpen())
# ar.write([0x32])
# ar.write([0x32])
# time.sleep(4)

def ar_send(mode, pos, degree, grip):
    mode = mode 
    pos = pos
    degree = degree
    grip = grip

    pos = int(pos).to_bytes(2, 'big')
    p_h = pos[0]
    p_l = pos[1]

    data = [0xFF, 0xFF, mode, p_h, p_l, degree, grip, 0xFF]
    ar.write(data)
    time.sleep(1)
    print(data)

def send_data(mode, hx, lx, hy, ly):

    Hx = hx
    Lx = lx
    Hy = hy
    Ly = ly
    mode = mode
    data = [0xFF, 0xFF, mode, Hx, Lx, Hy, Ly]
    ser.write(bytes(data))
    time.sleep(1)
    print(data)

def get_color():
    print("---------------get color---------------")
    time.sleep(1)
    send_data(0x0F, 0x00, 0x00, 0x00, 0x00)
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x00':
            break
    time.sleep(1)
    ar_send(0x0C, 290, 0, 0)
    while(1):
        mss = ar.read(3)
        print(mss)
        if mss == b'\xff\xff\x01':
            break
    time.sleep(1)
    ar_send(0x1C, 0, 0, 1)
    time.sleep(1)
    ar_send(0x0C, 190, 0, 0)
    while(1):
        mss = ar.read(3)
        print(mss)
        if mss == b'\xff\xff\x01':
            break
    time.sleep(1)
    x_start, y_start, x_list, y_list, x_end, y_end, z_list, z_start, z_end = color_detection()
    
    x_start = int(x_start).to_bytes(2, 'big')
    y_start = int(y_start).to_bytes(2, 'big')
    x_h = x_start[0]
    x_l = x_start[1]
    y_h = y_start[0]
    y_l = y_start[1]
    # print(x[0])
    # print(x[1])
    send_data(0x0B, x_h, x_l, y_h, y_l)
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x98':
            break
    ar_send(0x0C, z_start*10, 0, 0)
    while(1):
        mss = ar.read(3)
        print(mss)
        if mss == b'\xff\xff\x01':
            break
    time.sleep(1)
    
    for i in range(len(x_list)):
        time.sleep(1)
        x = int(x_list[i]).to_bytes(2, 'big')
        y = int(y_list[i]).to_bytes(2, 'big')
        x_h = x[0]
        x_l = x[1]
        y_h = y[0]
        y_l = y[1]
        z = z_list[i]*10

        time.sleep(1)
        ar_send(0x0C, z, 0, 0)
        while(1):
            mss = ar.read(3)
            print(mss)
            if mss == b'\xff\xff\x01':
                break
        time.sleep(1)

        send_data(0x0B, x_h, x_l, y_h, y_l)
        ser.flushInput()
        while(1):
            data = ser.read(3)
            print(data)
            if data == b'\xff\xff\x98':
                time.sleep(1)
                break
    time.sleep(1)
    x_end = int(x_end).to_bytes(2, 'big')
    y_end = int(y_end).to_bytes(2, 'big')
    xe_h = x_end[0]
    xe_l = x_end[1]
    ye_h = y_end[0]
    ye_l = y_end[1]
    # print(x[0])
    # print(x[1])
    ar_send(0x0C, z_end*10, 0, 0)
    while(1):
        mss = ar.read(3)
        print(mss)
        if mss == b'\xff\xff\x01':
            break
    time.sleep(1)
    send_data(0x0B, xe_h, xe_l, ye_h, ye_l)
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x98':
            break
    main()

def prepair_img():
    print("---------------prepairing img---------------")
    num = input("number of flip img: ")
    preparing_img(int(num))
    prepaing_template()  
    edge_detection()
    # polygon_detection_show()
    # detect_chess_show()
    main()

def get_Map2():
    print("---------------get Map2 (Flip)---------------")
    send_data(0x1A, 0, 0, 0, 0)
    get_map()
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x09':
            prepair_img()
                
def get_Map1():
    print("---------------get Map1---------------")
    send_data(0x1A, 0, 0, 0, 0)
    get_map()
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x09':
            a = input("Push 'a' to cap again(Flip): ")
            if a == 'a':
                get_Map2()
def get_template_map():
    print("---------------get template---------------")
    send_data(0x2A, 0, 0, 0, 0)
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x19':
            get_map()
            a = input("Push 'a' to get map: ")
            if a == 'a':
                get_Map1()
                
def procressing():
    get_template_map()
    
def sethome():
    print("---------------setting Home---------------")
    print("Sending....")
    send_data(0x0A, 0x00, 0x00, 0x00, 0x00)
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x99':
            break
    ar.write('s'.encode())
    main()
    
def go_pos():
    print("---------------Go Position---------------")
    x = input("X: ")
    y = input("Y:")
    x = int(x).to_bytes(2, 'big')
    y = int(y).to_bytes(2, 'big')
    x_h = x[0]
    x_l = x[1]
    y_h = y[0]
    y_l = y[1]
    # print(x[0])
    # print(x[1])
    send_data(0x0B, x_h, x_l, y_h, y_l)
    
    ser.flushInput()
    while(1):
        data = ser.read(3)
        print(data)
        if data == b'\xff\xff\x98':
            main()

def main():
    print("Program Start!")
    print("---------------MENU---------------")
    print("1.Set Home")
    print("2.Go to position")
    print("3.Start Procress!")
    print("4.let's Play")
    menu = input("ANS: ")
    if menu == '1':
        sethome()
    elif menu == '2':
        go_pos()
    elif menu == '3':
        procressing()       
    elif menu == '4':
        get_color()   
while(True):
    main()
# ar.write(bytes([30]))
# ar.write(0xFF)
# ar.write(('s').encode())