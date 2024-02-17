This repo showcases all of our experiments run in CSSE463
A majority of these are MATLAB files and can be run by changing the image directory to the location of the dataset
Dataset found here: https://github.com/Wangjing1551/LogoDet-3K-Dataset

For our final model, YOLOv5, we've included a small sample of the dataset only including the Maserati and Viper classes and you can run the validation on those images by entering the yolov5/ directory and then running the command
python val.py --task test --weights runs/train/exp23/weights/best.pt --data logoData.yaml --img 640
