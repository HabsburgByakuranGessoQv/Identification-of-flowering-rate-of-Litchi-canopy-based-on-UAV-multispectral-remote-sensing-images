'''
用于将所有的影像的NIR波段合成在在一个影像中（n,height,width）
'''
import os
import numpy as np
# python读取遥感影像，写出遥感影像
from osgeo import gdal


# os.environ['PROJ_LIB'] = r'e:\Anaconda3\envs\cloneTF21\Library\share\proj'
class GRID:

    # 读影像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        # 近红外波段
        im_data = dataset.GetRasterBand(1).ReadAsArray(0, 0, im_width, im_height)

        del dataset
        return im_data, im_width, im_height, im_geotrans, im_proj

    # 写成tif影像
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset


def get_file_names(data_dir, file_type=['tif', 'tiff']):
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = maindir + '/' + filename
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name


cet_list = ['0%', '10~20%', '50~60%', '80%']
classes = ["0%", "10%~20%", "50%~60%", "over_80%"]

# 单波段影像文件夹
in_dir1 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\R_tif/'
in_dir2 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\G_tif/'
in_dir3 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\B_tif/'
in_dir4 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\OSAVI/'
in_dir5 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\NDVI/'
in_dir6 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\NDRE/'
in_dir7 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\LCI/'
in_dir8 = r'D:\APPDATAiD\aliyunpan\CanopyFinal\GNDVI/'

# 输出文件夹
out_dir = r'D:\APPDATAiD\aliyunpan\CanopyFinal\Final\OSAVI_NDVI_NDRE_LCI\\'

files = []
datas = []
# listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
file_type = 'tif'
for i in range(len(cet_list)):
    oin_dir1 = os.path.join(in_dir1, cet_list[i])
    # print(oin_dir1)
    oin_dir2 = os.path.join(in_dir2, cet_list[i])
    oin_dir3 = os.path.join(in_dir3, cet_list[i])
    oin_dir4 = os.path.join(in_dir4, cet_list[i])
    oin_dir5 = os.path.join(in_dir5, cet_list[i])
    oin_dir6 = os.path.join(in_dir6, cet_list[i])
    oin_dir7 = os.path.join(in_dir7, cet_list[i])
    # oin_dir8 = os.path.join(in_dir8, cet_list[i])
    oout_dir = os.path.join(out_dir, classes[i])
    # 判断是否存在文件夹
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(oout_dir):
        os.mkdir(oout_dir)
    data_dir_list1, _ = get_file_names(oin_dir1, file_type)
    print(data_dir_list1)
    data_dir_list2, _ = get_file_names(oin_dir2, file_type)
    data_dir_list3, _ = get_file_names(oin_dir3, file_type)
    data_dir_list4, _ = get_file_names(oin_dir4, file_type)
    data_dir_list5, _ = get_file_names(oin_dir5, file_type)
    data_dir_list6, _ = get_file_names(oin_dir6, file_type)
    data_dir_list7, _ = get_file_names(oin_dir7, file_type)
    # data_dir_list8, _ = get_file_names(oin_dir8, file_type)

    for each_index, each_dir in enumerate(data_dir_list1):
        run = GRID()
        data1, height1, width1, geotrans1, proj1 = run.read_img(data_dir_list1[each_index])
        data2, height2, width2, geotrans2, proj2 = run.read_img(data_dir_list2[each_index])
        data3, height3, width3, geotrans3, proj3 = run.read_img(data_dir_list3[each_index])
        data4, height4, width4, geotrans4, proj4 = run.read_img(data_dir_list4[each_index])
        data5, height5, width5, geotrans5, proj5 = run.read_img(data_dir_list5[each_index])
        data6, height6, width6, geotrans6, proj6 = run.read_img(data_dir_list6[each_index])
        data7, height7, width7, geotrans7, proj7 = run.read_img(data_dir_list7[each_index])
        # data8, height8, width8, geotrans8, proj8 = run.read_img(data_dir_list8[each_index])
        datas.append(data1)
        datas.append(data2)
        datas.append(data3)
        datas.append(data4)
        datas.append(data5)
        datas.append(data6)
        datas.append(data7)
        # datas.append(data8)

        datas = np.array(datas)
        print(datas.shape)
        run.write_img(oout_dir + '/' + each_dir.split('/')[-1], proj1, geotrans1, datas)  # 输出文件命名
        datas = []
