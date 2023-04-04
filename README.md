# Joint Salient Object Detection and Camouflaged Object Detection via Uncertainty-aware Learning

![](https://github.com/baneitixiaomai/joint_sod_cod/blob/master/overview.png) 

This code is the journal extension version of: [* Uncertainty-aware Joint Salient Object and Camouflaged Object Detection *](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Uncertainty-Aware_Joint_Salient_Object_and_Camouflaged_Object_Detection_CVPR_2021_paper.pdf) (CVPR2021)

## Train the model
### Set up
```ruby
pip install -r requirements.txt  
```
### Train Model
- Prepare data for training (We provided the related data in:[[Google_Drive]](https://drive.google.com/drive/folders/1J_B7mNEwB0ryzngd8JyZdllfFNgkiFdh?usp=sharing). Please download it and put it in the '/train_data/' folder)   
  - 'duts+wmae' means the augmented SOD training set  
  - 'COD_train'  means the COD training dataset  
  - 'JPEGImages_select' means the selected PASCAL VOC dataset  
```ruby
python train.py  
```
###  Test and Evaluate Model
- Prepare data for testing (We provided the related SOD and COD test image and ground-truth in:[[Google_Drive]](https://drive.google.com/drive/folders/1-yRpkCm2d7qKjW01tGCfmbHL5zqMeBI1). Please download it and put it in the '/test_data/' folder)  

```ruby
python test.py   
```
## Pretrained model and Prediction Maps
### Trained model:
Please download the trained model and put it in "./models/": [[Google_Drive]](https://drive.google.com/drive/folders/1-2rk7k1GeeeWQvQOmJeecm9E6k0rz2fH?usp=sharing);

### Prediction Maps:
Results of our model on four benchmark datasets (CAMO, CHAMELEON, COD10K, NC4K) for COD:[[Google_Drive]](https://drive.google.com/file/d/1hPSPEZHBCYYti3Sw_YVau1teE1lePmH8/view?usp=sharing); and six benchmark datasets (DUTS, ECSSD, DUT, HKU-IS, PASCAL, SOD ) for SOD: [[Google_Drive]](https://drive.google.com/file/d/1kMBp0ZUyxJUCtZzhoVhXp3B_fA9WvL-x/view?usp=sharing)
 
#### Thanks Yuxin Mao for the Evaluation code [link](https://github.com/fupiao1998/SOD-Eval)
