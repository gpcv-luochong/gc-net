# gc-net

#implement gc-net(Geometry and Context Network) by pytorch

1.pip安装requirements.txt中需要的环境

opencv_contrib_python_headless==3.4.3.18

torchvision==0.2.1

torch==0.4.0

matplotlib==2.2.2

numpy==1.14.3

tensorboardX==1.4

2.需要下载KITTI2015数据集，对应的read_data.py文件只能读取KITTI2015数据集

3.KITTI2015数据集下载需要翻墙

4.更改main.py中KITTI2015数据集的路径

5.运行main.py（训练时主要就三个文件read_data.py,network.py,main.py）
