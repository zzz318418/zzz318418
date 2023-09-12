import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import pyrebase
import os
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import eel
import sys
import time
import numpy as np
import cv2
import os
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
import random
import string
import cgi
import time
import hashlib

PORT = 8000  # 伺服器埠號
FILE_PATH = "D:/download/output_video_with_audio.mp4"  # 要傳送的MP4檔案路徑


class RequestHandler(BaseHTTPRequestHandler):
    count = 0
    def __init__(self, *args, **kwargs):
        self.timeout = 200
        super().__init__(*args, **kwargs)

    # @classmethod
    # def get_self(self):
    #     return self.path




    def do_GET(self):

        if self.path == '/vdo':
            self.send_response(200)
            self.send_header('Content-type', 'video/mp4')
            self.end_headers()

            with open(FILE_PATH, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()





    def do_POST(self):
        # 接收POST請求的body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        print("Received selectedData from Android:", post_data)

        # 分割字串並取出四個字串
        data_parts = post_data.split(',')
        if len(data_parts) ==4:
            concatenated_string = ''.join(data_parts)
            # 使用 MD5 雜湊生成 8 位元的字母
            # config = {
            #     "apiKey": "AIzaSyASJIbbFK7dbTACZ8ccpy0M9AxadSxkAjQ",
            #     "authDomain": "lorlan-71f7f.firebaseapp.com",
            #     "databaseURL": "https://lorlan-71f7f-default-rtdb.firebaseio.com",
            #     "projectId": "lorlan-71f7f",
            #     "storageBucket": "lorlan-71f7f.appspot.com",
            #     "messagingSenderId": "259398933893",
            #     "appId": "1:259398933893:web:b2e19001900352267cb8ec",
            #     "measurementId": "G-CR8TY3V8X4"
            # }
            #
            # firebase = pyrebase.initialize_app(config)
            # storage = firebase.storage()
            # my_image = "pic (6).jpg"
            #
            # # firebase storage
            # # Upload Image
            # # storage.child("pic1.jpg").put("pic1.jpg")
            # import os
            # # Download Image
            # storage.child(my_image).download(
            #     filename="download.jpg", path=os.path.basename(my_image))
            #
            # # get firestore data
            # cred = credentials.Certificate("lorlan-71f7f-firebase-adminsdk-ulwy3-162d9c28c3.json")
            # firebase_admin.initialize_app(cred)
            # db = firestore.client()
            #
            # while True:
            #
            #     doc_ref = db.collection(u'users').document(u'person1')
            #     # doc_ref = db.collection(u'Users').document(u'person2')
            #
            #     doc = doc_ref.get()
            #     if doc.exists:
            #         print(f'Document data: {doc.to_dict()}')
            #         storage.child(my_image).download(
            #             filename="download.jpg", path=os.path.basename(my_image))
            #         break
            #     else:
            #         print(u'No such document!')
            #
            #     # time.sleep(2)
            #
            # print("next move")
            #
            # # ----------------------------------------firebase------------------------------------------------------------------------------
            #
            # # ----------------------------------------辨識和去背------------------------------------------------------------------------------
            #
            # import torch
            # import cv2
            # import numpy as np
            # import os
            # import numpy as np
            # import torch
            # import matplotlib.pyplot as plt
            # import cv2
            # # import ultralytics
            # # from ultralytics import YOLO
            # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            # from IPython.display import display, Image
            # import cv2
            # import numpy as np
            # import matplotlib.pyplot as plt
            #
            # # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/best.pt',
            #                        force_reload=True)
            # # model.conf = 0.7
            # # print(model)
            # img_first = cv2.imread('download.jpg')
            # results = model(img_first)
            # results.print()
            # print(results.xyxy)
            #
            # # cv2.imshow('YOLO COCO', np.squeeze(results.render()))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #
            # def show_mask(mask, ax, random_color=False):
            #     if random_color:
            #         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            #     else:
            #         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            #     h, w = mask.shape[-2:]
            #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            #     ax.imshow(mask_image)
            #
            # def show_points(coords, labels, ax, marker_size=375):
            #     pos_points = coords[labels == 1]
            #     neg_points = coords[labels == 0]
            #     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size,
            #                edgecolor='white',
            #                linewidth=1.25)
            #     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size,
            #                edgecolor='white',
            #                linewidth=1.25)
            #
            # def show_box(box, ax):
            #     x0, y0 = box[0], box[1]
            #     w, h = box[2] - box[0], box[3] - box[1]
            #     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
            #
            # def class_name(num):
            #     if num == 0:
            #         return "title"
            #     if num == 1:
            #         return "vice - title"
            #     if num == 2:
            #         return "brand"
            #     if num == 3:
            #         return "name"
            #     if num == 4:
            #         return "prodself"
            #     if num == 5:
            #         return "price"
            #     if num == 6:
            #         return "date"
            #     if num == 7:
            #         return "date"
            #     if num == 8:
            #         return "others"
            #
            # print(results.xyxy[0])
            #
            # result_c = results.pandas().xyxy[0].to_dict(orient="records")
            # i = 1
            # ct = 0
            # for result in result_c:
            #     con = result['confidence']
            #     cs = result['class']
            #     x1 = int(result['xmin'])
            #     y1 = int(result['ymin'])
            #     x2 = int(result['xmax'])
            #     y2 = int(result['ymax'])
            #     # Do whatever you want
            #     # print(x1,y1,x2,y2)
            #     cut = cv2.imread("download.jpg")
            #
            #     # 裁切區域的 x 與 y 座標（左上角）
            #     x = x1
            #     y = y1
            #
            #     # 裁切區域的長度與寬度
            #     w = x2 - x1
            #     h = y2 - y1
            #     print(y1, y2, x1, x2)
            #     # print(x,y,w,h)
            #     # print(y,y + h, x,x + w)
            #     # print(x, y, x+w, y+h)
            #
            #     bbox = [x, y, x + w, y + h]
            #     # 111 465 461 817
            #     # 124 278 584 380
            #
            #     sam_checkpoint = "sam_vit_b_01ec64.pth"
            #     model_type = "vit_b"
            #     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            #     predictor = SamPredictor(sam)
            #
            #     image = cv2.cvtColor(cv2.imread('download.jpg'), cv2.COLOR_BGR2RGB)
            #     predictor.set_image(image)
            #
            #     input_box = np.array(bbox)
            #
            #     masks, _, _ = predictor.predict(
            #         point_coords=None,
            #         point_labels=None,
            #         box=input_box[None, :],
            #         multimask_output=False,
            #     )
            #
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(image)
            #     show_mask(masks[0], plt.gca())
            #     # show_box(input_box, plt.gca())
            #     plt.axis('off')
            #     plt.show()
            #
            #     segmentation_mask = masks[0]
            #     binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
            #
            #     white_background = np.ones_like(image) * 255
            #
            #     new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
            #
            #     name = class_name(cs)
            #
            #     plt.imshow(new_image.astype(np.uint8))
            #     plt.axis('off')
            #     plt.savefig('good_ad/src_element/temp_' + name + '.png',
            #                 bbox_inches='tight',
            #                 pad_inches=0,
            #                 transparent=True,
            #                 format='png',
            #                 )
            #     plt.show()
            #
            #     import sqlite3
            #     from tkinter import Tk, Button, Canvas
            #     from PIL import Image, ImageFont
            #
            #     # 如果当前位深是32的话，可以不用写转RGBA模式的这一句，但是写上也没啥问题
            #     # 从RGB（24位）模式转成RGBA（32位）模式
            #     img = Image.open('good_ad/src_element/temp_' + name + '.png').convert('RGBA')
            #     W, L = img.size
            #     white_pixel = (255, 255, 255, 255)  # 白色
            #     for h in range(W):
            #         for i in range(L):
            #             if img.getpixel((h, i)) == white_pixel:
            #                 img.putpixel((h, i), (0, 0, 0, 0))  # 设置透明
            #
            #     img.save('good_ad/src_element/' + name + '.png')  # 自己设置保存地址
            #     ct += 1
            #     file_name = 'C:/Users/user/PycharmProjects/final_final/good_ad/src_element/temp_' + name + '.png'
            #     final_source = 'good_ad/src_element/' + name + '.png'
            #     image = Image.open(final_source)
            #     bbox = image.getbbox()
            #     if bbox:
            #         print("Bounding Box Coordinates (left, upper, right, lower):", bbox)
            #
            #         y1 = int(bbox[1])
            #         y2 = int(bbox[3])
            #         x1 = int(bbox[0])
            #         x2 = int(bbox[2])
            #         x = x1
            #         y = y1
            #         w = x2 - x1
            #         h = y2 - y1
            #
            #         crop_source = cv2.imread(final_source)
            #         crop_img = crop_source[y:y + h, x:x + w]
            #         cv2.imwrite("good_ad/src_element/" + name + ".png", crop_img)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            #         # img = Image.open('good_ad/src_element/' + name + '.png').convert('RGBA')
            #         # W, L = img.size
            #         # white_pixel = (0, 0, 0, 100)  # 白色
            #         # for h in range(W):
            #         #     for i in range(L):
            #         #         if img.getpixel((h, i)) == white_pixel:
            #         #             img.putpixel((h, i), (0, 0, 0, 0))  # 设置透明
            #         #
            #         # img.save('good_ad/src_element/' + name + '.png')  # 自己设置保存地址
            #     else:
            #         print("No bounding box found for the image.")
            #
            #     os.remove(file_name)
            # ----------------------------------------辨識和去背------------------------------------------------------------------------------
            # # -------------------------------------------video-----------------------------------------------------------------------------

            import eel
            import sys
            import time
            import numpy as np
            import cv2
            import os
            import subprocess
            import threading

            # @eel.expose
            # def get_self():
            #     # RequestHandler.get_s = classmethod(RequestHandler.get_self)
            #     # self_value = RequestHandler()
            #     # self_value=self_value.get_self
            #     self_value=RequestHandler.get_self()
            #     # self_value = RequestHandler.get_s
            #
            #
            #
            #     print("22222222222self.path ="+self_value)
            #     if self_value == '/vdo':
            #         self.send_response(200)
            #         self.send_header('Content-type', 'video/mp4')
            #         self.end_headers()
            #
            #         with open(FILE_PATH, 'rb') as f:
            #             self.wfile.write(f.read())
            #     else:
            #         self.send_response(404)
            #         self.end_headers()



            def number_of_file():
                dir = "D:/download/canvas_screen_shoot"  # chrome的download的資料夾位置
                initial_count = 0
                for path in os.listdir(dir):
                    if os.path.isfile(os.path.join(dir, path)):
                        initial_count += 1
                return initial_count

            def merge_video_and_audio(video_file, audio_file, output_file):
                # 使用 ffmpeg 命令合併影片和聲音
                cmd = f'ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {output_file}'
                subprocess.run(cmd, shell=True)

            @eel.expose
            def add_music():
                print("i can add music")
                # 使用範例
                video_file = r'D:/download/test.mp4'
                audio_file = r'tikms18.mp3'
                output_file = r'D:/download/output_video_with_audio.mp4'
                merge_video_and_audio(video_file, audio_file, output_file)

            @eel.expose
            def makevedio():
                # 確認圖片尺寸(每張都要相同)
                size = (800, 1400)
                # 確認圖片數量
                time.sleep(5)
                quantity = number_of_file()
                print("number_of_png:", quantity)
                # 完成寫入物件的建立，第一個引數是合成之後的影片的名稱，第二個引數是可以使用的編碼器，第三個引數是幀率即每秒鐘展示多少張圖片，第四個引數是圖片大小資訊
                videowrite = cv2.VideoWriter(r'D:/download/test.mp4', -1, 30,
                                             size)  # 路徑必須不包含中文 #D:\download\canvas_screen_shoot
                img_array = []
                for filename in [r'D:/download/canvas_screen_shoot/screenshot{0}.jpg'.format(i) for i in
                                 range(1, quantity)]:  # D:\download\canvas_screen_shoot
                    img = cv2.imread(filename)
                    if img is None:
                        print(filename + " is error!")
                        continue
                    img_array.append(img)
                # 寫入影片
                for i in range(1,
                               (
                               quantity)):  # for i in range(1,quantity-3)才不會IndexError: list index out of range?我也不知道為什麼:(
                    videowrite.write(img_array[i])
                print('Finish')

            # 要檢查的檔案路徑
            brand_png = "good_ad/src_element/brand.png"
            name_png = "good_ad/src_element/name.png"
            others_png = "good_ad/src_element/others.png"
            price_png = "good_ad/src_element/price.png"
            prodself_png = "good_ad/src_element/prodself.png"
            title_png = "good_ad/src_element/title.png"
            png = [brand_png, name_png, others_png, price_png, prodself_png, title_png]
            # 等待所有圖片完成生成
            count = 0
            import itertools

            counter1 = itertools.count()

            def number_of_images_generated():
                # global count
                count = next(counter1)

                if os.path.isfile(png[count]):
                    print(png[count], "存在。")
                    count += 1
                return count

            def check_image_quantity(target_quantity):
                while True:
                    current_quantity = number_of_images_generated()  # 假設有一個函式可以獲取當前圖片數量
                    if current_quantity >= target_quantity:
                        break
                    time.sleep(1)  # 等待一秒後再次檢查


            # @eel.expose
            # def after_eel():
            #     self.send_response(200)
            #     self.send_header('Content-type', 'text/plain')
            #     self.end_headers()
            #     self.wfile.write(response_data.encode('utf-8'))

            import threading
            import eel

            # 其他的 import 和函式定義...

            def run_eel():
                # 設定目標圖片數量
                target_quantity = 6
                # 開始檢查圖片數量
                check_image_quantity(target_quantity)
                # 當圖片數量達到目標數量後，繼續執行下面的程式碼
                print("Image quantity reached the target, continue...")
                eel.init('good_ad')
                print('Calling Javascript...')
                eel.start('module.html', size=(800, 500))


            if __name__ == '__main__':
                eel_thread = threading.Thread(target=run_eel)
                eel_thread.start()

                # 在這裡可以繼續執行其他程式碼，不會被 Eel 事件循環阻塞
                print("Other code after starting Eel...")

                # 等待 Eel 執行緒結束
                eel_thread.join()

                # 在這裡可以執行一些在 Eel 執行緒結束後的後續處理
                response_data = "SUCCESS"
        else:
            response_data = "ERROR"
        time.sleep(10)
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))

        # ----------------------------------------firebase------------------------------------------------------------------------------















def run_server():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Serving at port {PORT}")
    httpd.serve_forever()





run_server()








#===============================================製作影片並回傳================================================================================

@eel.expose
def transport():
    run_server()

































