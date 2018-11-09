# flowers_detection
## 简介
本项目为基于迁移学习的花朵多分类;尝试的思路有如下四个:
1.参考leNet搭建简单网络,acc 80%
(train_leNet.py)
2.利用vgg16提取瓶颈层特征,接softmax层分类 acc 80%
(vgg_feature.py train_vgg.py)
3.利用resnet50,xception,InceptionV3提取瓶颈层融合特征,接softmax分类 acc 94%
(gap.py gap_train.py predict)
4.在2.的基础上放开最后几层进行finetune
(train_vgg_finetune.py)

## 声明
1.本项目属于新手练手,理论上,1,2,3,4步骤的acc会逐步提高,但实际上3.的acc最高(~94%),有时间会继续调整
2.cam.ipynb 实现CAM(类激活映射),参考链接如下

## 参考

https://github.com/ypwhs/dogs_vs_cats
https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/image_classification_using_very_little_data/