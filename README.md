The task was image segmentation for ship detection.

For begin was held a minor data analysis, its represent in the Data_Exploratory.ipynb file.
It was determined that the number of images with ships is 22%. And the number of ships in the image varies from 1 to 15 (in most cases 1).
Becous task for the model is to determine the pixels that represent the ship. It should be noted that these pixels are less than 0.001% of the total pixels in all data. In conclusion we can notice that the data is imbalance.

The model learning process is presented in TrainModel.ipynb | trainmodel.py files.
Data augmentation was introduced to solve the problem of file imbalance. The architecture for the neural network has been chosen U-net.
Used metrics dice coef, binary accuracy. The loss function is gdice coef loss.

In the main.py file, this model has the ability to work in 3 modes. To process one image, to process all images in a folder, and to permanently process images in a specific folder.
To process one image, you need to run through the command python main.py image path_to_image.
To process all images in a folder, you need to run a command python main.py folder path_to_folder.
And for to constantly monitor and process all images in the folder, you need to execute the command python main.py stream path_to_folder.

