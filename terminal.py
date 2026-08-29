import os
import time
import csv
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import evaluate
import train
import preprocess

DIRS = ["C:\\Users\\super\\Documents\\projects\\Yayscires",
        ["diversification", ["set_1", ["chips_G"], ["chips_NIR"], ["chips_R"], ["chips_RE"], ["chips_RGB"]], ["set_2", ["chips_G"], ["chips_NIR"], ["chips_R"], ["chips_RE"], ["chips_RGB"]]],
        ["evaluate", ["stats"]],
        ["new_data", ["chips", ["chips_G"], ["chips_NIR"], ["chips_R"], ["chips_RE"], ["chips_RGB"]], ["labeled", ["labeled_G"], ["labeled_NIR"], ["labeled_R"], ["labeled_RE"], ["labeled_RGB"]], ["raw_data"]],
        ["prior_data", ["Data", ["aligned"]]]]

evaluate.set_consts(DIRS)
train.set_consts(DIRS)
preprocess.set_consts(DIRS)

def wheeee():
    myImage = Image.open(os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0], DIRS[3][1][5], "chip_rgb20.jpg"))
    img = myImage.copy()
    array = np.asarray(img)
    if array.dtype != np.uint8:
        array = (array - array.min()) / (array.max() - array.min()) * 255
        array = array.astype(np.uint8)
    image = Image.fromarray(array, mode='L')
    array = np.stack((array, array, array), axis=-1)

    img = Image.fromarray(array, mode="RGB")
    img.show()
    myImage.show()

def init(folder_path):
    print("Clearing subfolders...")
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, subfolder))]

    for subfolder in subfolders:
        print(f"Clearing subfolder: {subfolder}        ", end = "\r")
        for item in os.listdir(subfolder):
            item_path = os.path.join(subfolder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    os.rmdir(item_path)  # Remove empty folder
            except Exception as e:
                print(f"Error clearing {item_path}: {e}", end = "\r")

    print("\nFinding CSV files...", end = "\r")
    csv_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
                 if file.endswith('.csv') and os.path.isfile(os.path.join(folder_path, file))]
    print(f"Found {len(csv_paths)}")
    print("Trimming CSV files to the first entry...")
    for csv_path in csv_paths:
        try:
            print(f"Processing CSV", end = "\r")
            df = pd.read_csv(csv_path)
            if not df.empty:
                x = df.iloc[:0]
                with open(csv_path, 'w') as file:
                    pass
                x.to_csv(csv_path, index=False)
            else:
                print(f"CSV {csv_path} is empty, skipping.", end = "\r")
        except Exception as e:
            print(f"Error processing CSV {csv_path}: {e}", end = "\r")

    print("Completed clearing subfolders and CSVs.                                                                           ")

def create_folder():
    main_folder = os.path.join(DIRS[0], DIRS[2][0])
    sub_folder = os.path.join(main_folder, DIRS[2][1][0])
    os.makedirs(sub_folder, exist_ok=True)
    print(f"Folders created: {main_folder} and {sub_folder}")
    for csv_file in [f'test_labels_{band}.csv' for band in ['G','NIR','R','RE','RGB']]:
        csv_path = os.path.join(main_folder, csv_file)
        df = pd.DataFrame(columns=['filename','xmin','ymin','xmax','ymax','class'])
        df.to_csv(csv_path, index=False)
        print(f"Created: {csv_path}")

def diversity_testing():
    sample_size = 100
    scaling_factors, noise_params = evaluate.find_diversity_params(sample_size)
    folder_path = os.path.join(DIRS[0], DIRS[1][0], DIRS[1][1][0])                                  # make sure this is set to the right set within diversification
    subfolders = [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, subfolder))]
    for subfolder in subfolders:
        print(f"Clearing subfolder: {subfolder}", end = "\r")
        for item in os.listdir(subfolder):
            item_path = os.path.join(subfolder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Remove file
                elif os.path.isdir(item_path):
                    os.rmdir(item_path)  # Remove empty folder
            except Exception as e:
                print(f"Error clearing {item_path}: {e}", end = "\r")


    evaluate.analyze_dataset_similarity(folder_path, ['G', 'NIR', 'R', 'RE', 'RGB'], sample_size)
    
    preprocess.diversify(folder_path, scaling_factors, noise_params, sample_size)
    sanity_regression()

def sanity_regression(band):
    # Prepare data for a single band
    print(f"preparing model on labels_{band}.csv", end = "\r")
    root = os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0])

    data = pd.read_csv(os.path.join(root, f"labels_{band}.csv"))
    X_G, y_G = evaluate.prepare_flattened_data(data, os.path.join(root, f'chips_{band}'))

    # Train a logistic regression model
    X_train, X_test, y_train, y_test = train_test_split(X_G, y_G, test_size=0.3, train_size=0.7, random_state=42)
    print("training model                   ", end = "\r")
    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(X_train, y_train)

    # Evaluate the model
    print("evaluating model", end = "\r")
    y_pred = baseline_model.predict(X_test)
    print("Baseline Accuracy:", accuracy_score(y_test, y_pred))

def make_modified():
    for band in ['G', 'NIR', 'R', 'RE', 'RGB']:
        indf = pd.read_csv(os.path.join(DIRS[0], DIRS[2][0], f'test_{band}.csv'))
        labels = pd.read_csv(os.path.join(DIRS[0], DIRS[3][0], 'chips', f'labels_{band}.csv'))
        outdf = pd.read_csv(os.path.join(DIRS[0], DIRS[2][0], f'modified_{band}.csv'))

        images = []
        for idx in range(1161):
            name = indf.iloc[idx,0]
            if name not in images:
                images.append(name)
        for name in images:
            matching_rows = labels[labels.iloc[:, 0] == name]
            outdf = pd.concat([outdf, matching_rows], ignore_index=True)
        
        outdf.to_csv(os.path.join(DIRS[0], DIRS[2][0], f'modified_{band}.csv'), index=False)
    
def test_preprocessing():
    preprocess.test_preprocessing()

def do_preprocessing():
    init(os.path.join(DIRS[0], DIRS[3][0], DIRS[3][1][0]))
    init(os.path.join(DIRS[0], DIRS[2][0]))
    preprocess.jitter()
    preprocess.diversify()
    preprocess.prep()

def test_training(band, bs):
    train.do_training(band, bs, test=[0.1,50], path=os.path.join(DIRS[0], DIRS[2][0], f'b{bs}_pth', f'model_{band}.pth'))
    evaluate.do_evaluation(band, bs, test=0.1, path=os.path.join(DIRS[0], DIRS[2][0], f'b{bs}_pth', f'model_{band}.pth')) 

def do_training():
    for bs in [32]:
        for band in ['G','NIR','R','RE','RGB']:            
            try:
                train.do_training(band, bs)
            except Exception as e:
                print(f"failed: {e}")
                
def test_evaluation():
    evaluate.do_evaluation('G', 32, test=0.2, path=None, t=0)

def do_evaluation(bs, thresh=None):
    for band in ['G', 'NIR', 'R', 'RE', 'RGB']:        
        #x = input(f"Press [enter] to start evaluating Model_{band}, [s] to skip, [b] to break ")
        x = ''
        if (x.lower() == "c"):
            continue
        elif (x.lower() == "b"):
            break
        evaluate.do_evaluation(band, bs, test=None, path=None, t=thresh)

def quick_test(band='G', bs=16, test_split=0.1, num_epochs=5, threshold=None):
    """
    Quickly test the training and evaluation pipeline on a small dataset and limited epochs.

    Args:
        band (str): The spectral band to test. Default is 'G'.
        bs (int): Batch size for training and evaluation. Default is 8.
        test_split (float): Fraction of the dataset to use for testing. Default is 0.1 (10%).
        num_epochs (int): Number of epochs to train. Default is 5.
        threshold (float): Confidence threshold for evaluation. Default is 0.5.
    """
    trainm = False
    testm = True
    print(f"Starting quick test for band {band} with batch size {bs} and {num_epochs} epochs.")

    if trainm:
        # Train the model
        print("Training...")
        train.do_training(
            band=band,
            bs=bs,
            test=[test_split, num_epochs],  # Specify test split and epochs
            path=os.path.join(DIRS[0], DIRS[2][0], f'b{bs}_pth', f'model_quick_test_{band}.pth')
        )
    
    if testm:
        # Evaluate the model
        print("Evaluating...")
        evaluate.do_evaluation(
            band=band,
            bs=bs,
            test=test_split,  # Specify test split
            path=os.path.join(DIRS[0], DIRS[2][0], f'b{bs}_pth', f'model_quick_test_{band}.pth'),
            t=threshold,  # Confidence threshold
            wannasee = True
        )
    
    print(f"Quick test for band {band} completed successfully.")

quick_test(test_split = 1, num_epochs = 300, threshold = 0.24)
