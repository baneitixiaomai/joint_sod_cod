## Joint Salient Object Detection and Camouflaged Object Detection via Uncertainty-aware Learning

![](https://github.com/baneitixiaomai/joint_sod_cod/blob/master/overview.png)  
## Set up
- pip install -r requirements.txt  

## Train Model
- Prepare data for training (We provided the related data in:[[train_data]](). Please download it and put it in the '/train_data/' folder) 
  -- 'duts+wmae' means the augmented SOD training set  
  -- 'COD_train'  means the COD training dataset  
  -- 'JPEGImages_select' means the selected PASCAL VOC dataset  
- Run train.py   

##  Test Model
- Run ./test.py  

## Trained model:
Please download the trained model and put it in "./models/": [[Google_Drive]](https://drive.google.com/drive/folders/1PYb-1EKooiXW2KZ_IWwVRAzCYKhmcSn8?usp=sharing);

##  Prediction Maps
Results of our model on four benchmark datasets for COD (CAMO, CHAMELEON, COD10K, NC4K), and six benchmark datasets (DUTS, ECSSD, DUT, HKU-IS, PASCAL, SOD ) for SOD, which can be found: [[Google_Drive]](https://drive.google.com/file/d/1q8Ai6U0O61R4b1wDPeF1h2UN42X9W0KJ/view?usp=sharing)
 
