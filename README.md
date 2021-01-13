# objectdetection
object detection by the pytorch


now a days the computer vision is every where so the decetion and analysis of the objects and the persions pasiing in the video frames is  more difficult for the normal algarithms  because of the  intensity and thershold values differrence in the frames 


so the headach can be simply solved by the deeplearning because of the deep learning uses he automatic feauture learning algarithms 



in this project we can use the retinanet _fpn with the resnet backbone   with the pre trained model the reina net is based on the focal loss and the pyramid based structure opertaions on the images 

we can use the open cv for the image operations 


the most important parameter we used here is the thershold  which can incraeses or dicreases the prediction classes 


you can use the foloowing comands to get the results



git clone https://github.com/suddulavenkatanaresh/objectdetection_pytorch.git


cd /suddulavenkatanaresh/objectdetection_pytorch


# you canb use the default thershold value 


python detect_images.py --input <input image path>


 #you can chise your own values of thersold and the min-size values 

python detect_images.py --input <image_path>    --min-size 1200 --threshold 0.5



or else if you want to  make the detection on the videos 


python detect_videos.py --input input/video2.mp4





