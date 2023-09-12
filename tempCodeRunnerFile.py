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
