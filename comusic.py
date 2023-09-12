import subprocess

def merge_video_and_audio(video_file, audio_file, output_file):
    # 使用 ffmpeg 命令合併影片和聲音
    cmd = f'ffmpeg -i "{video_file}" -i "{audio_file}" -c:v copy -c:a aac -strict experimental "{output_file}"'
    subprocess.run(cmd, shell=True)

# 使用範例
video_file = r'D:/download/test.mp4'
audio_file = r'D:/download/tikms18.mp3'
output_file = r'D:/download/output_video_with_audio.mp4'
merge_video_and_audio(video_file, audio_file, output_file)
