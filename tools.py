import datetime
import cv2
import os
import time
import shutil

# getting special_suffix files list
def flat_list(the_list):
    now = the_list[:]
    res = []
    while now:
        head = now.pop(0)
        if isinstance(head, list):
            now = head +now
        else:
            res.append(head)
    return res
def get_files(Folder):
    path = []
    for file in os.listdir(Folder):
        if  not os.path.isdir(Folder + '/' + file):
            path.append(Folder + '/' + file)
        else:
            path.append(get_files(Folder + '/' + file))
    return flat_list(path)
def get_specified_files(Folder, Suffix):
    Files = get_files(Folder)
    Files_Selected = []
    L = len(Suffix)
    for File in Files:
        Temp = File[- 1 *L:]
        if Temp == Suffix:
            Files_Selected.append(File)
    return Files_Selected

def get_folders_name(root):
    names = []
    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            names.append(name)
    return names

#get absolute_path of all folders under root
def get_folders_path(root):
    paths = []
    for root, dirs, files in os.walk(root, topdown=False):
        for name in dirs:
            paths.append(os.path.join(root, name))
    return paths

#delete empty folders under root
def delete_empty_folders(root):
    if os.path.isdir(root):
        for d in os.listdir(root):
            delete_empty_folders(os.path.join(root, d))
        if not os.listdir(root):
            os.rmdir(root)
#copy folder
def copy_folder(folder_from, folder_to):
    if os.path.exists(folder_to) == False:
        os.mkdir(folder_to)
    for item in os.listdir(folder_from):
        s = os.path.join(folder_from, item)
        d = os.path.join(folder_to, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, False, None)
        else:
            shutil.copy2(s, d)

# get_all_images_under_folder
def get_image_names(folder):
    image_names = []
    bmp_files = get_specified_files(folder, ".bmp")
    image_names = image_names + bmp_files
    jpg_files = get_specified_files(folder, ".jpg")
    image_names = image_names + jpg_files
    png_files = get_specified_files(folder, ".png")
    image_names = image_names + png_files

    return image_names
def get_delta_date(start_date, delta):
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    date_with_delta =(start_date + datetime.timedelta(days=delta)).strftime("%Y%m%d")
    return date_with_delta

def get_today():
    today = datetime.date.today()
    date_today = today.strftime('%Y%m%d')
    return date_today

def get_toyear():
    today = datetime.date.today()
    toyear = today.year
    return toyear

def get_date_list(start_date, end_date):
    list = []
    datestart = datetime.datetime.strptime(start_date, '%Y%m%d')
    dateend = datetime.datetime.strptime(end_date, '%Y%m%d')
    list.append(datestart.strftime('%Y%m%d'))
    while datestart < dateend:
        datestart += datetime.timedelta(days=1)
        list.append(datestart.strftime('%Y%m%d'))
    return  list

def standard_images(folder, size, suffix):
    image_names = get_image_names(folder)
    for image_name in image_names:
        image = cv2.imread(image_name)
        image = cv2.resize(image, size)
        os.remove(image_name)
        suffix_this = image_name.split('.')
        suffix_this = suffix_this[-1]
        image_name = image_name.replace(suffix_this, suffix)
        print(image_name)
        cv2.imwrite(image_name, image)

#6个季度以内有效
def get_season_border(trade_date, k):
    year = trade_date[0] + trade_date[1] + trade_date[2] + trade_date[3]
    month = trade_date[4] + trade_date[5]
    season = int((int(month)-1)/3)
    season_border0 = trade_date
    season_border1 = trade_date
    if season == 0:
        season_border0 = year + '0101'
        season_border1 = year + '0331'
    if season == 1:
        season_border0 = year + '0401'
        season_border1 = year + '0630'
    if season == 2:
        season_border0 = year + '0701'
        season_border1 = year + '0930'
    if season == 3:
        season_border0 = year + '1001'
        season_border1 = year + '1231'
    bias_list = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15]
    season_border0 = get_delta_date(season_border0, k * 90)
    season_border1 = get_delta_date(season_border1, k * 90)
    for i in range(len(bias_list)):
        season_border0_try = get_delta_date(season_border0, bias_list[i])
        if (season_border0_try[-1] == '1') and (season_border0_try[-2] == '0'):
            season_border0 = season_border0_try
            break
    for i in range(len(bias_list)):
        season_border1_try = get_delta_date(season_border1, bias_list[i])
        if (season_border1_try[-1] == '0') and (season_border1_try[-2] == '3'):
            season_border1 = season_border1_try
        if (season_border1_try[-1] == '1') and (season_border1_try[-2] == '3'):
            season_border1 = season_border1_try
            break
    return season_border0, season_border1

if __name__ == '__main__':
    # border_list = get_season_border_list('20020101')
    date0, date1 = get_season_border('20020304', -1)
    print(date0)
    print(date1)