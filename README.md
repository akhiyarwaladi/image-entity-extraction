# image-entity-extraction

This project is under development.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. We use anaconda environment with python3.6, all requirement all listed below.
```
pip install -r requirements.txt
```

### Prerequisites

Besides code, you have to download external requirement from s3 Bucket. Execute this command or simply download from s3 GUI and extract in same folder as your code. AWS Access Key ID and Secret Access Key is required.


#### Model
Model that we use contain pretrained model for face detection (tiny face, ssd mobilenet), face recognition (out custom dataset embedding) and age gender prediction for unknown person (on progress)
```
aws s3 cp s3://akhiyar-bucket/image_entity_extraction/Models/ Models/ --recursive
```

#### Dataset
```
aws s3 cp s3://akhiyar-bucket/face_train_processed/ face_train_processed/ --recursive

```
Put your anotated dataset in root directory of this project inside ./Dataset/Entity/raw

if you plan to make your own dataset you can anotate using labelimg. Preprocessing to make your data ready to embed follow and run all jupyter notebook CropROI.ipynb, TakeFaceSample.ipynb and split_train_test.ipynb. Each will crop all face to match facenet format and split into balance train test data.

#### Align face
```
python src/align_dataset_mtcnn.py ./Dataset/Entity/raw ./Dataset/Entity/processed --image_size 160 --margin 0 --random_order --gpu_memory_fraction 0.25
```
This align code will allow you to skip annotation step, using face mtcnn alignment method. Make sure person in images data only the label that you want. Otherwise will be false anotated.

#### Train / Embed 
```
python src/classifier.py TRAIN ./Dataset/Entity/processed ./Models/facenet/20180402-114759.pb ./Models/Entity/Entity.pkl --batch_size 1000
```

#### Test
```
python src/faceRec.py --path ../../image_entity/images/test7.jpg

```