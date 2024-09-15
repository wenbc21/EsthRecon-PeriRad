## 该文件夹是用来存放训练数据的目录
### 使用步骤如下：
* （1）在dataset文件夹下创建新文件夹"flower_data"
* （2）点击链接下载花分类数据集 [https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
* （3）解压数据集到flower_data文件夹下
* （4）可以自己划分训练集、验证集和测试集。在flower_photos下创建train、val和test，把每一类中的图片放到train、val和test下对应的类中
* （5）运行train.py时，指定data_path参数为数据集路径（如dataset/flower_data）

```
├── flower_data   
       ├── flower_photos（解压的数据集文件夹，3670个样本）  
       ├── train（训练集）  
       └── val（验证集） 
       └── test（测试集） 
```
