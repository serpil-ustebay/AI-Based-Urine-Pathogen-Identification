Project Title:# AI-Based-Urine-Pathogen-Identification
Rapid Identification of Escherichia coli and Klebsiella pneumoniae Species in Urine Cultures: using Artificial Intelligence-Assisted Photo Processing


Author: Serpil Ustebay
Email: serpil.ustebay@gmail.com
GitHub: https://github.com/serpil-ustebay/AI-Based-Urine-Pathogen-Identification

Project Description
This project aims to develop an AI-based image processing system for rapid and accurate identification of Escherichia coli and Klebsiella pneumoniae in urine culture images. The model analyzes bacterial colonies in petri dish images and performs classification accordingly.

 Requirements
To run this project, the following Python packages are required:

numpy
opencv-python
tensorflow
scikit-learn
matplotlib
yolov12
ultralytics
wandb

Model Weights (best.pt)
The file best.pt contains the trained weights of the YOLOv12m model specifically trained for this project. It stores the learned parameters after training on the petri dish bacteria colony images, enabling the model to detect and classify E. coli and K. pneumoniae colonies accurately.You can use this file to run inference directly without retraining the model.

To train the model:
python TrainModel.py


To classify a single image:
python ClassifyImage.py --image_path "path_to_image.jpg"


The model's performance was evaluated using the following metrics:

Accuracy
Precision
Recall
F1 Score
Specificity
False Positive Rate (FPR)
False Negative Rate (FNR)
Hamming Loss
