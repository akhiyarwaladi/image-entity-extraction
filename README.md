# image-entity-extraction

This project is under development.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Besides code, you have to download external requirement from s3 Bucket. Execute this command or simply download from s3 GUI and extract in same folder as your code


#### Dataset
```
aws s3 cp s3://akhiyar-bucket/face_train_processed/ face_train_processed/ --recursive

```
Put your anotated dataset in root directory of this project inside ./Dataset/Entity/raw


#### Align face
```
python src/align_dataset_mtcnn.py ./Dataset/Entity/raw ./Dataset/Entity/processed --image_size 160 --margin 0 --random_order --gpu_memory_fraction 0.25
```

#### Train / Embed 
```
python src/classifier.py TRAIN ./Dataset/Entity/processed ./Models/facenet/20180402-114759.pb ./Models/Entity/Entity.pkl --batch_size 1000
```

#### Test
```
python src/faceRec.py --path ../../image_entity/images/test7.jpg

```