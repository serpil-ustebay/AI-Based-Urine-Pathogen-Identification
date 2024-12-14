import cv2
import pandas as pd
import os
from sklearn.metrics import classification_report, hamming_loss, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

def showPredictedImage(model, thresh=0.0, save=False,savedname=""):
    source=choose_file()
    print(source)
    results = model.predict(source)
    result = results[0]

    image = cv2.imread(source)
    df_predicted_objects = pd.DataFrame(columns=['id', 'prob'])

    # check all bound box
    for box in result.boxes:
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        class_id = result.names[box.cls[0].item()]
        conf = round(box.conf[0].item(), 2)
        if conf > thresh:
            cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), (255, 0, 0), 2)
            image = cv2.putText(image, "ID:" + str(class_id) + ', Prob:' + str(conf), (cords[0], cords[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            df_predicted_objects = df_predicted_objects._append({"id": class_id, "prob": conf}, ignore_index=True)

    # Display image using Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    fig1 = plt.gcf()
    plt.show()
    if save:
        fig1.savefig(savedname, dpi=300)



def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file chooser dialog
    file_path = filedialog.askopenfilename()

    return file_path


def classify_bacteria(source_image, show=False):
    # source = Source_Path + "54218385516101_W23_T960.JPG"
    results = model.predict(source_image)
    result = results[0]

    image = cv2.imread(source_image)
    df_predicted_objects = pd.DataFrame(columns=['id', 'prob'])

    for box in result.boxes:
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        class_id = result.names[box.cls[0].item()]
        # print(class_id)
        conf = round(box.conf[0].item(), 2)
        cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), (255, 0, 0), 2)
        image = cv2.putText(image, "Bacteria:" + str(class_id), (cords[0], cords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA)
        df_predicted_objects = df_predicted_objects._append({"id": class_id, "prob": conf}, ignore_index=True)

    if df_predicted_objects.shape[0] == 0:
        print("Bacteria tanımlanamadı: ", source_image)
        return -1
    else:
        value_counts = df_predicted_objects['id'].value_counts()
        classified_bacteria = value_counts.idxmax()
        return classified_bacteria


def list_jpg_files(directory, file_end='.jpg'):
    jpg_files = [file for file in os.listdir(directory) if file.endswith(file_end)]
    return jpg_files

def predict_TestData_Calculate_PM():
    predictions = pd.DataFrame(columns=['file', 'true_y', 'pred_y'])

    Source_Path = "..\\Data\\Test Data\\1\\"
    jpg_files = list_jpg_files(Source_Path)
    for file in jpg_files:
        source = Source_Path + file
        predictions = predictions._append({"file": file, 'true_y': 1, "pred_y": classify_bacteria(source)},
                                          ignore_index=True)
    print("Bacteria 1 are predicted.")

    Source_Path = "..\Data\\Test Data\\2\\"
    jpg_files = list_jpg_files(Source_Path)

    for file in jpg_files:
        source = Source_Path + file
        predictions = predictions._append({"file": file, 'true_y': 2, "pred_y": classify_bacteria(source)},
                                          ignore_index=True)
    print("Bacteria 2 are predicted.")

    print("Prediction değerleri ", predictions['pred_y'].value_counts())
    print("Sınıflandırılamayan Örnek sayısı:\n ", predictions[predictions['pred_y'] == -1])

    # Eğer var sınıflandırılamayan örnek var ise drop edilerek cm çıkarılacak
    predictions.drop(predictions.loc[predictions['pred_y'] == -1].index, inplace=True)

    # Save DataFrame to a CSV file
    predictions.to_csv('predictions.csv', index=False)

    y_true = predictions["true_y"].to_numpy()
    y_pred = predictions["pred_y"].to_numpy()
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Extract TP, TN, FP, FN from confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # Calculate specificity and false positive rate
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)

    # False negative rate
    fnr = fn / (fn + tp)

    # Calculate Hamming score
    hamming_loss_value = hamming_loss(y_true, y_pred)

    # Print the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Specificity:", specificity)
    print("False Positive Rate (FPR):", fpr)
    print("False Negative Rate (FNR):", fnr)
    print("Hamming loss:", hamming_loss_value)
    print("Classification Report", classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(cm)

    # Define class names
    class_names = ['1', '2']

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2])
    disp.plot(cmap='Blues')
    fig1 = plt.gcf()
    plt.show()

    fig1.savefig('confusion_matrix_last.jpg', dpi=300)



######################################

if __name__ == '__main__':
    best = r'C:\Users\serpi\PycharmProjects\IDSScada\runs\detect\train\weights\best.pt'
    model = YOLO(best)  # load a pretrained model (recommended for training)
    showPredictedImage(model,thresh=0.8,save=True,savedname="predictedlast.jpg")
