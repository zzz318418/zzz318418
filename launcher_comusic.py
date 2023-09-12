import eel
import sys
import time
import numpy as np
import cv2
import os
import subprocess
import threading


def number_of_file():
    dir = "D:/download/canvas_screen_shoot"     #chrome的download的資料夾位置
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
    print ("i can add music")
    # 使用範例
    video_file = r'D:/download/test.mp4'
    audio_file = r'D:/download/tikms18.mp3'
    output_file = r'D:/download/output_video_with_audio.mp4'
    merge_video_and_audio(video_file, audio_file, output_file)
    



@eel.expose
def makevedio():  
    # 確認圖片尺寸(每張都要相同)
    size = (800,1400)
    # 確認圖片數量
    time.sleep(5)
    quantity = number_of_file()
    print("number_of_png:",quantity)
    # 完成寫入物件的建立，第一個引數是合成之後的影片的名稱，第二個引數是可以使用的編碼器，第三個引數是幀率即每秒鐘展示多少張圖片，第四個引數是圖片大小資訊
    videowrite = cv2.VideoWriter(r'D:/download/test.mp4',-1,30,size) # 路徑必須不包含中文 #D:\download\canvas_screen_shoot
    img_array=[]
    for filename in [r'D:/download/canvas_screen_shoot/screenshot{0}.jpg'.format(i) for i in range(1,quantity)]:#D:\download\canvas_screen_shoot
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)
    # 寫入影片
    for i in range(1,(quantity)):#for i in range(1,quantity-3)才不會IndexError: list index out of range?我也不知道為什麼:(
        videowrite.write(img_array[i])
    print('Finish')
    









# 要檢查的檔案路徑
brand_png = "good_ad/src_element/brand.png"
name_png = "good_ad/src_element/name.png"
others_png = "good_ad/src_element/others.png"
price_png = "good_ad/src_element/price.png"
prodself_png = "good_ad/src_element/prodself.png"
title_png = "good_ad/src_element/title.png"
png=[brand_png,name_png,others_png,price_png,prodself_png,title_png]
# 等待所有圖片完成生成
count=0
def number_of_images_generated():
    global count
    if os.path.isfile(png[count]):
        print(png[count],"存在。")
        count+=1
    return count

def check_image_quantity(target_quantity):
    while True:
        current_quantity = number_of_images_generated()  # 假設有一個函式可以獲取當前圖片數量
        if current_quantity >= target_quantity:
            break
        time.sleep(1)  # 等待一秒後再次檢查

# 設定目標圖片數量
target_quantity = 6

# 開始檢查圖片數量
check_image_quantity(target_quantity)

# 當圖片數量達到目標數量後，繼續執行下面的程式碼
print("Image quantity reached the target, continue...")
#在這個範例中，check_image_quantity 函式會在一個循環中持續檢查當前生成的圖片數量是否達到目標數量。如果是，則跳出循環，並允許程式繼續執行下面的程式碼。
#請注意，您需要根據實際情況自行實現 number_of_images_generated 函式，以獲取當前生成的圖片數量。此外，如果您擔心無限循環可能造成效能問題，您可以在循環中添加適當的等待時間（如 time.sleep），以降低 CPU 使用率。
eel.init('good_ad')
print('Calling Javascript...')
#eel.my_javascript_function(1, 2, 3, 4)     # This calls the Javascript function
eel.start('module.html', size=(800, 500))   #eel.start('module+.html', mode=None)

