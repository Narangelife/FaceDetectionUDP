import cv2
import socket

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    M_SIZE = 1024

    host = '127.0.0.1'
    port = 25561

    clia = ('127.0.0.1', 25562)

    localAddress = (host, port)
    sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
    print('Socket Created')
    sock.bind(localAddress)

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

        s = ''

        # 検出した顔に印を付ける
        for (x, y, w, h) in face_list:
            color = (0, 0, 225)
            pen_w = 3
            cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness = pen_w)
            s = '{}_{}_{}_{}'.format(x, y, w, h)

        # UDP通信
        if s != '':
            print(s)
            sock.sendto(s.encode(encoding='utf-8'), clia)

        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
    sock.close()
