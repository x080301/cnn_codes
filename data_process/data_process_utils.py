import imgaug.augmenters as iaa
import numpy as np
import cv2
import os
import shutil
import random


def Load(dir, ob_train=False):
    # 遍历每个文件
    file_name_list = os.listdir(dir)

    i = 0

    ob_das_erst_ist = True
    ob_das_50st_ist = True

    for file_name in file_name_list:

        if ob_train:

            if (i % 50) == 0:
                print(i)
                print("\n")
                0

        # 读取图片
        image = cv2.imread(dir + file_name)

        # 放缩一下<----暂时
        if not (image.shape == (32, 32)):
            image = cv2.resize(image, (32, 32))

        image_label = file_name.split('_')[0]

        # image_label = image_label.split("000")[1]

        image_label = int(image_label)

        # 拓展维度
        image = np.expand_dims(image, axis=0)

        '''
        #去掉一些数据<-暂时
        if image_label < 9:
            image_label = image_label
        elif image_label == 9:
            continue
        elif image_label < 19:
            image_label -= 1
        elif image_label == 19:
            continue
        elif image_label < 25:
            image_label -= 2
        elif image_label == 25:
            continue
        elif image_label < 53:
            image_label -= 3
        elif image_label == 53:
            continue
        else:
            image_label -= 4
        '''

        # 构造矩阵
        if i % 100 == 0:

            if ob_das_erst_ist:
                ob_das_erst_ist = False
            elif ob_das_50st_ist:
                image_label_ndarray_0 = image_label_ndarray
                image_ndarrary_0 = image_ndarrary
                ob_das_50st_ist = False
            else:
                image_label_ndarray_0 = np.hstack((image_label_ndarray_0, image_label_ndarray))
                image_ndarrary_0 = np.vstack((image_ndarrary_0, image_ndarrary))

            image_label_ndarray = np.array(image_label)
            image_ndarrary = np.array(image)

        else:

            image_label_ndarray = np.hstack((image_label_ndarray, image_label))
            image_ndarrary = np.vstack((image_ndarrary, image))
        i += 1
    image_label_ndarray_0 = np.hstack((image_label_ndarray_0, image_label_ndarray))
    image_ndarrary_0 = np.vstack((image_ndarrary_0, image_ndarrary))
    return image_ndarrary_0, image_label_ndarray_0


# image_ndarrary, label_ndarry=Load('C:/Users/Lenovo/Desktop/cnn/program/other_data_set/TSRD_data_set/tsrd-train/')
# print(label_ndarry)
# print(np.shape(image_ndarrary))


def update(input_img_path, output_img_path, lightness, saturation):
    """
    用于修改图片的亮度和饱和度
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    :param lightness: 亮度
    :param saturation: 饱和度
    """
    # 调整最大值
    MAX_VALUE = 100
    # 加载图片 读取彩色图像归一化且转换为浮点型
    image = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # 1.调整亮度（线性变换)
    hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # 饱和度
    hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(np.uint8)
    cv2.imwrite(output_img_path, lsImg)


def light_change(dataset_dir, output_dir, wievielemal=5):
    # 这里调参！！！
    # lightness = -50#random.randint(-80,-50)#int(input("lightness(亮度-100~+100):")) # 亮度
    saturation = -50  # int(input("saturation(饱和度-100~+100):")) # 饱和度

    # 删除原来的
    for file_name in os.listdir(output_dir):
        os.remove(output_dir + '/' + file_name)

    # 获得需要转化的图片路径并生成目标路径

    j = 0

    for i in range(wievielemal):

        file_name_list = os.listdir(dataset_dir)

        for file_name in file_name_list:
            output_img_path = output_dir + file_name.split("_")[0] + "_" + str(j) + ".jpg"

            input_img_path = dataset_dir + file_name

            j = j + 1

            update(input_img_path, output_img_path, random.randint(-80, -10), saturation)


def image_augment(input_dir, augmented_dir, coefficient=10):
    # 删除原来的
    for file_name in os.listdir(augmented_dir):
        os.remove(augmented_dir + '/' + file_name)

    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.2, 0.4)),  # 高斯模糊
        # iaa.Add((-50,50), per_channel=0.5),#为图像中的所有像素添加值
        iaa.Sometimes(0.3, iaa.AddElementwise((-40, 40))),  # 相邻像素加法增强器
        iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=0.5)),  # 高斯噪声
        iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=0.03)),  # 扭曲增强器
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)),  # 像素位移增强器
        # iaa.Sometimes(0.2, iaa.Superpixels(p_replace=0.05, n_segments=8)) #部分地将图像变换为其超像素表示
    ], random_order=True)  # do all of the above in random order

    i = 0

    for file_name in os.listdir(input_dir):

        image = cv2.imread(input_dir + file_name)

        if not (image.shape == (64, 64)):
            image = cv2.resize(image, (64, 64))

        images_aug = [seq.augment_image(image) for _ in range(coefficient)]  # 应用data augmentation
        # ia.imshow(ia.draw_grid(images_aug, cols=4, rows=2))  # 显示图片操作效果

        for img in images_aug:
            output_file_name = file_name.split("_")[0] + "_" + str(i) + ".jpg"

            cv2.imwrite(augmented_dir + output_file_name, img)

            i += 1


def ppm_2_png(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        image = cv2.imread(input_dir + file_name)
        cv2.imwrite(output_dir + file_name + '.png', image)


def save_as(folder_dir, des_dir, scale=7):
    # 删除原来的
    for file_name in os.listdir(des_dir + 'valid/'):
        os.remove(des_dir + 'valid/' + file_name)
    for file_name in os.listdir(des_dir + 'train/'):
        os.remove(des_dir + 'train/' + file_name)
    print('removed')

    file_name_list = os.listdir(folder_dir)

    index_list = [0] * 43

    for file_name in file_name_list:

        image_label = file_name.split('_')[0]

        image_label = int(image_label)

        if (index_list[image_label] % scale) == 0:
            des_folder_dir = des_dir + 'valid/'
        else:
            des_folder_dir = des_dir + 'train/'

        index_list[image_label] += 1

        shutil.copy(folder_dir + file_name, des_folder_dir)
        os.rename(des_folder_dir + file_name, des_folder_dir + "tt0000" + file_name)


def save_as_pretraining(folder_dir, des_dir, scale=7):
    # 删除原来的
    for file_name in os.listdir(des_dir + 'test/'):
        os.remove(des_dir + 'test/' + file_name)
    for file_name in os.listdir(des_dir + 'valid/'):
        os.remove(des_dir + 'valid/' + file_name)
    for file_name in os.listdir(des_dir + 'train/'):
        os.remove(des_dir + 'train/' + file_name)
    print('removed')

    file_name_list = os.listdir(folder_dir)

    index_list = [0] * 43

    for file_name in file_name_list:

        if file_name[0] == 'G':
            continue

        image_label = file_name.split('_')[0]

        image_label = image_label.split("000")[1]

        image_label = int(image_label)

        if (index_list[image_label] % scale) == 0:
            des_folder_dir = des_dir + 'test/'
        elif (index_list[image_label] % scale) == 1:
            des_folder_dir = des_dir + 'valid/'
        else:
            des_folder_dir = des_dir + 'train/'

        index_list[image_label] += 1

        shutil.copy(folder_dir + file_name, des_folder_dir)


import pickle


def pickle_dump(set_type, data_to_dump, output_dir):
    file_name = output_dir + set_type + '.p'

    with open(file_name, 'wb') as f:
        pickle.dump(data_to_dump, f)


def image_dump(input_dir,output_dir):

    print('dump start')
    test = {}
    X_test, y_test = Load(input_dir + 'test/')
    test['features'] = X_test
    test['labels'] = y_test
    pickle_dump('test', test, output_dir)
    print('test done')

    valid = {}
    X_valid, y_valid = Load(input_dir + 'valid/')
    valid['features'] = X_valid
    valid['labels'] = y_valid
    pickle_dump('valid', valid, output_dir)
    print('valid done')

    X_train, y_train = Load(input_dir + 'train/')
    train = {}
    train['features'] = X_train
    train['labels'] = y_train
    pickle_dump('train', train, output_dir)
    print('train done')


class DataGrouping:
    class4pretraining = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 25, 31, 33, 35, 38]
    saved_m = [20, 0, 1, 2, 3, 4, 21, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 14, 15, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
               33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

    def get_map(self):

        a = self.class4pretraining

        al = len(a)

        b = [-1] * 43
        j1 = 0
        j2 = 0
        for i in range(43):
            if (i == a[j1]) and (i < len(a)):
                b[i] = j1
                j1 += 1
            else:
                b[i] = j2 + 20
                j2 += 1
        return b

    def raw_data(self, folder_dir, des_dir):

        # 参数
        scale = 6

        # 映射关系
        m = self.get_map()

        # 删除原来的
        for file_name in os.listdir(des_dir + 'test/'):
            os.remove(des_dir + 'test/' + file_name)
        for file_name in os.listdir(des_dir + 'valid/'):
            os.remove(des_dir + 'valid/' + file_name)
        for file_name in os.listdir(des_dir + 'train/'):
            os.remove(des_dir + 'train/' + file_name)
        print('removed')

        folder_name_list = os.listdir(folder_dir)

        # 遍历文件夹

        for folder_name in folder_name_list:

            # 遍历每个文件
            file_name_list = os.listdir(folder_dir + folder_name)

            random.shuffle(file_name_list)
            i = 0
            class_id = m[int(folder_name)]
            for file_name in file_name_list:

                # 读取合适图片
                if file_name[0] == 'G':
                    continue

                file_dir = folder_dir + folder_name + '/' + file_name

                if int(folder_name) == 42:
                    continue
                elif (i % scale) == 0:
                    des_folder_dir = des_dir + 'test/'
                elif (i % scale) == 1:
                    des_folder_dir = des_dir + 'valid/'
                else:
                    des_folder_dir = des_dir + 'train/'
                i += 1

                image = cv2.imread(file_dir)
                cv2.imwrite(des_folder_dir + str(class_id) + '_' + str(i) + '.png', image)

            print(folder_name + '   done')
