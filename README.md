# flowers_detection
## 简介
本项目为基于迁移学习的花朵多分类,尝试的思路如下:

* 参考leNet搭建简单网络,acc 80%  
```
train_leNet.py  
```

* 利用vgg16提取瓶颈层特征,接softmax层分类 acc 80%  
```
vgg_feature.py train_vgg.py  
```

* 利用resnet50,xception,InceptionV3提取瓶颈层融合特征,接softmax分类 acc 94%  
```
gap.py gap_train.py predict.py  
```

* 在vgg16的基础上放开最后几层进行finetune  
```
train_vgg_finetune.py  
```

* 实现CAM(类激活映射)  
```
cam.ipynb
```

## 声明
1. 本项目属于新手练手,理论上,以上步骤的acc会逐步提高,但实际上第三个的acc最高(94%),有时间会继续调整
2. 迁移学习有很多技巧,入门推荐kaggle的猫狗分类,参考链接如下

## 参考

 https://github.com/ypwhs/dogs_vs_cats
 https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/image_classification_using_very_little_data/