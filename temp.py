import os
path = '/home/piai/github/Yolo_mark/x64/Release/data/img'

file_list = os.listdir(path)

jpg_list = [file for file in file_list if file.endswith('.jpg')]
txt_list = [file for file in file_list if file.endswith('.txt')]

len(jpg_list) # 858
len(txt_list) # 862 -> 4개 데이터 삭제필요 확인

result = []
for txt in txt_list:
    txt = txt[:-4]
    if txt not in map(lambda x: x[:-4], jpg_list):
        result.append(txt)

# 4개 데이터 목록 생성
result # ['16483_Mask', '17268_Mask', '17270_Mask', '17123_Mask']
to_be_deleted = list(map(lambda x: x+'.txt', result))

# 삭제
for file in to_be_deleted:
    os.remove(img_path+file)

# 폴더내 이미지파일 이름 확인하여 train.txt만들기
fw_dir = 'x64/Release/data/img/'
# jpg_list

with open('/media/piai/SAMSUNG/labeling_data/mask_new_data_02.25/with_mask1/train.txt', 'w') as f:
    for jpg in jpg_list:
        line = fw_dir + jpg
        f.write(line+"\n")