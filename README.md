The task was image segmentation for ship detection.

For begin was held a minor data analysis, its represent in the Data_Exploratory.ipynb file.
It was determined that the number of images with ships is 22%. And the number of ships in the image varies from 1 to 15 (in most cases 1).
Becous task for the model is to determine the pixels that represent the ship. It should be noted that these pixels are less than 0.001% of the total pixels in all data. In conclusion we can notice that the data is imbalance.

The model learning process is presented in G_dice.ipynb file.
Data augmentation was introduced to solve the problem of file imbalance. The architecture for the neural network has been chosen U-net.
Used metrics dice coef, binary accuracy, true positive rate. The loss function is gdice coef loss.
For all attempts of training this architecture, the maximum dice score was 0.82. Which gave Private score: 0.76566 result on the kaggle.
In general, the obtained result is very small, and the neural network notices everything as not a ship. Which gives a good result for binary accuracy, but very small for true positive rate.
Other architectures and approaches have been used. And the result was either similar, or on the contrary, the neural network noticed every object that is not water like a ship (for example, clouds or stones). As solution you can then classify the observed objects with other СNN models, and this will have a good accuracy. But it was not set in the condition of solving the task.

In the main.py file, this model has the ability to work in 3 modes. To process one image, to process all images in a folder, and to permanently process images in a specific folder.
To process one image, you need to run through the command python main.py image path_to_image.
To process all images in a folder, you need to run a command python main.py folder path_to_folder.
And for to constantly monitor and process all images in the folder, you need to execute the command python main.py stream path_to_folder.
No error handling is currently implemented

In tools.py file consists functions that need for neural network work.
