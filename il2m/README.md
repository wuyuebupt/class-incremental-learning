# IL2M-Class-Incremental-Learning-with-Dual-Memory
## Abstract
This paper presents a class incremental learning (IL) method which exploits fine tuning and a dual memory to reduce the negative effect of catastrophic forgetting in image recognition. 

First, we simplify the current fine tuning based approaches which use a combination of classification and distillation losses to compensate for the limited availability of past data. We find that the distillation term actually hurts performance when a memory is allowed. Then, we modify the usual class IL memory component. 

Similar to existing works, a first memory stores exemplar images of past classes.
A second memory is introduced here to store past class statistics obtained when they were initially learned. 
The intuition here is that classes are best modeled when all their data are available and that their initial statistics are useful across different incremental states. 

A prediction bias towards newly learned classes appears during inference because the dataset is imbalanced in their favor.
The challenge is to make predictions of new and past classes more comparable. To do this, scores of past classes are rectified by leveraging contents from both memories.

The method has negligible added cost, both in terms of memory and of inference complexity.
Experiments with three large public datasets show that the proposed approach is more effective than a range of competitive state-of-the-art methods. 
## Paper
[[pdf]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)-[[supp]](http://openaccess.thecvf.com/content_ICCV_2019/supplemental/Belouadah_IL2M_Class_Incremental_ICCV_2019_supplemental.pdf)

To cite this work:

```
@InProceedings{Belouadah_2019_ICCV,
author = {Belouadah, Eden and Popescu, Adrian},
title = {IL2M: Class Incremental Learning With Dual Memory},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
} 
```

## Data
Data needed to reproduce the experiments from the paper is available [here](https://drive.google.com/open?id=1kgoB0Oxb9Wv2wSWFT5Yf7IoKXR3gAL_3)

## Requierements
* Python 2.7
* Pytorch 1.0.0
* Numpy 1.13.0
* SkLearn 0.19.1


## How to run

1. ### Training the first batch of classes from scratch (better on GPU)

```
python codes/scratch.py configs/scratch.cf
```

2. ### IL with Fine tuning (better on GPU)

```
python codes/ft.py configs/ft.cf
```
3. ### Scores extraction (better on GPU)

```
python codes/features_extraction_b1.py configs/features_extraction_b1_train.cf
python codes/features_extraction_b1.py configs/features_extraction_b1_val.cf
python codes/features_extraction_ft.py configs/features_extraction_ft.cf
```
4. ### IL2M (requires CPU only)
You should provide the following parameters to the program: images_list_files_path, scores_path, b1_scores_path, dataset_name, S, P, K

For example, for IL2M on ILSVRC1000 with 9 incremental states (10 states in total, each one having a number of classes = 100), with memory size 20000:
```
python codes/il2m.py  data/images_list_files/ /set/here/your/path/feat_scores_extract_for_ft/ /set/here/your/path/feat_scores_extract_for_first_batch/ ilsvrc 10 100 20000 2>&1 | tee /set/here/your/path/logs/il2m/ilsvrc/S~10/il2m_ilsvrc_s10_20k.log
```


### Remarks. 
1. If your dataset is different from ILSVRC, VGG-Face2 and Google Landmarks, you need to compute the images mean/std used for normalization of your dataset using the traing images of the first batch of classes and add it to the file 'data/datasets_mean_std.txt'.
2. Please delete all the comments from the configuration files, to avoid compilation errors. 
3. Feel free to send an email to eden.belouadah@cea.fr if there is any issue with the code.


### Updates 2020 Jan 06
1. Update the dataset using the REAL validation set of imagenet for validation

Results:

First batch: Val : acc@1 = 77.52% ; acc@5 = 92.02%

Results after several times break-resume:
```
top1 accuracies so far : [77.52, 67.95, 59.16, 53.40, 48.11, 44.26, 41.88, 39.34, 37.82, 35.12]
top5 accuracies so far : [92.02, 87.33, 82.32, 78.30, 74.79, 71.85, 70.96, 68.82, 66.66, 63.98]
```

Some auto-generated results
```
top1 accuracies so far : [44.26, 41.88857142857143, 39.345, 37.824444444444445, 35.152]
top5 accuracies so far : [71.85666666666667, 70.96857142857142, 68.825, 66.66222222222223, 63.984]
TOP1 validation accuracies = [44.26, 41.888, 39.345, 37.824, 35.152]
TOP5 validation accuracies = [71.856, 70.968, 68.825, 66.662, 63.984]
```

After il2m, results are:
```
Top1 Acc = [0.7752, 0.6795, 0.5916, 0.534, 0.4811, 0.4425, 0.4188, 0.3934, 0.3782, 0.3515]
Top5 Acc = [0.9201, 0.8733, 0.8232, 0.783, 0.7479, 0.7185, 0.7096, 0.6882, 0.6666, 0.6398]
Mean inc Acc | acc@1 = 0.4745 | acc@5 = 0.7389
***********************************************************************
*********************************IL2M**********************************
Top1 Acc = [0.7752, 0.7023, 0.6328, 0.594, 0.5383, 0.5031, 0.4887, 0.4681, 0.4349, 0.4091]
Top5 Acc = [0.9201, 0.883, 0.8373, 0.8087, 0.7665, 0.7356, 0.7291, 0.7078, 0.6801, 0.6544]
Mean inc Acc | acc@1 = 0.5301 | acc@5 = 0.7558
```
