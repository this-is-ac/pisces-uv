import os

# 1

# folder_name = '/home/aditya/llve/StableLLVE/outputs/PHiSHNet_01_UWVideo'
# file_name = 'phishnet'

# folder_name = '/home/aditya/llve/StableLLVE/outputs/PHiSHVideo_01_K10_UWVideo'
# file_name = 'phishvideo'

# folder_name = '/home/aditya/llve/StableLLVE/uw_video/images'
# file_name = 'original'

# folder_name = '/home/aditya/llve/StableLLVE/uw_video/train'
# file_name = 'bsr'

# 2

# folder_name = '/home/aditya/llve/StableLLVE/outputs/newvid_PhishVideo'
# file_name = '2_phishvideo'

# folder_name = '/home/aditya/llve/StableLLVE/uw_video_2/images'
# file_name = '2_original'

folder_name = '/home/aditya/llve/StableLLVE/uw_video_2/train'
file_name = '2_bsr'

os.system("ffmpeg -framerate 30 -pattern_type glob -i '{}/*.png' -c:v libx264 -pix_fmt yuv420p {}.mp4".format(folder_name, file_name))