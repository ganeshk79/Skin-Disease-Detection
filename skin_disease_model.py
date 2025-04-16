import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                         precision_score, recall_score, f1_score, roc_curve, auc, 
                         precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from PIL import Image
import json
from datetime import datetime

# Constants
IMG_SIZE = 224
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 50

def create_model():
    """
    Creates an ensemble model combining EfficientNetB3 and ResNet50
    """
    # Input layer
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # EfficientNetB3 branch
    efficientnet = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in efficientnet.layers[:-20]:  # Freeze early layers
        layer.trainable = False
    x1 = GlobalAveragePooling2D()(efficientnet.output)
    x1 = BatchNormalization()(x1)
    
    # ResNet50 branch
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in resnet.layers[:-20]:  # Freeze early layers
        layer.trainable = False
    x2 = GlobalAveragePooling2D()(resnet.output)
    x2 = BatchNormalization()(x2)
    
    # Combine both models
    combined = concatenate([x1, x2])
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # Output layer
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    return model

def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img

def create_data_generators(X_train, X_val, y_train, y_val):
    """
    Create data generators with augmentation for training
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE
    )
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator, class_weights=None):
    """
    Train the model with callbacks and class weights
    """
    # Callbacks
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    return history

def plot_roc_curves(y_test, y_pred_proba, class_names):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(12, 8))
    
    # Store AUC scores
    auc_scores = {}
    
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores[class_names[i]] = auc_score
        
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('evaluation_results/roc_curves.png')
    plt.close()
    
    return auc_scores

def plot_precision_recall_curves(y_test, y_pred_proba, class_names):
    """Plot Precision-Recall curves for each class"""
    plt.figure(figsize=(12, 8))
    
    # Store average precision scores
    ap_scores = {}
    
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
        ap_score = average_precision_score(y_test[:, i], y_pred_proba[:, i])
        ap_scores[class_names[i]] = ap_score
        
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap_score:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('evaluation_results/precision_recall_curves.png')
    plt.close()
    
    return ap_scores

def analyze_errors(y_test_classes, y_pred_classes, y_pred_proba, class_names):
    """Analyze most common misclassifications and uncertain predictions"""
    error_analysis = {
        'most_common_misclassifications': [],
        'high_uncertainty_predictions': []
    }
    
    # Find most common misclassifications
    for true_class in range(len(class_names)):
        for pred_class in range(len(class_names)):
            if true_class != pred_class:
                mask = (y_test_classes == true_class) & (y_pred_classes == pred_class)
                count = mask.sum()
                if count > 0:
                    error_analysis['most_common_misclassifications'].append({
                        'true_class': class_names[true_class],
                        'predicted_class': class_names[pred_class],
                        'count': int(count),
                        'confidence': float(y_pred_proba[mask, pred_class].mean())
                    })
    
    # Sort by count
    error_analysis['most_common_misclassifications'].sort(key=lambda x: x['count'], reverse=True)
    
    # Find predictions with high uncertainty (low confidence)
    max_probas = np.max(y_pred_proba, axis=1)
    uncertain_idx = np.where(max_probas < 0.8)[0]  # threshold can be adjusted
    
    for idx in uncertain_idx:
        error_analysis['high_uncertainty_predictions'].append({
            'true_class': class_names[y_test_classes[idx]],
            'predicted_class': class_names[y_pred_classes[idx]],
            'confidence': float(max_probas[idx]),
            'all_probas': {class_names[i]: float(y_pred_proba[idx, i]) 
                          for i in range(len(class_names))}
        })
    
    return error_analysis

def evaluate_model(model, X_test, y_test, class_names):
    # Create directory for evaluation results
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate basic metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test_classes, y_pred_classes)),
        'precision_macro': float(precision_score(y_test_classes, y_pred_classes, average='macro')),
        'precision_weighted': float(precision_score(y_test_classes, y_pred_classes, average='weighted')),
        'recall_macro': float(recall_score(y_test_classes, y_pred_classes, average='macro')),
        'recall_weighted': float(recall_score(y_test_classes, y_pred_classes, average='weighted')),
        'f1_macro': float(f1_score(y_test_classes, y_pred_classes, average='macro')),
        'f1_weighted': float(f1_score(y_test_classes, y_pred_classes, average='weighted'))
    }
    
    # Print overall metrics
    print("\n=== Overall Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nMacro-averaged metrics (treating all classes equally):")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall: {metrics['recall_macro']:.4f}")
    print(f"F1-score: {metrics['f1_macro']:.4f}")
    
    # Get and plot ROC curves and AUC scores
    print("\n=== ROC Curves and AUC Scores ===")
    metrics['auc_scores'] = plot_roc_curves(y_test, y_pred_proba, class_names)
    for class_name, auc_score in metrics['auc_scores'].items():
        print(f"{class_name}: {auc_score:.4f}")
    
    # Get and plot Precision-Recall curves
    print("\n=== Precision-Recall Curves and Average Precision Scores ===")
    metrics['average_precision_scores'] = plot_precision_recall_curves(y_test, y_pred_proba, class_names)
    for class_name, ap_score in metrics['average_precision_scores'].items():
        print(f"{class_name}: {ap_score:.4f}")
    
    # Confusion Matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Plot percentage confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Percentage)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix_percent.png')
    plt.close()
    
    # Error Analysis
    print("\n=== Error Analysis ===")
    error_analysis = analyze_errors(y_test_classes, y_pred_classes, y_pred_proba, class_names)
    metrics['error_analysis'] = error_analysis
    
    # Print top misclassifications
    print("\nTop 5 Most Common Misclassifications:")
    for error in error_analysis['most_common_misclassifications'][:5]:
        print(f"True: {error['true_class']}, Predicted: {error['predicted_class']}")
        print(f"Count: {error['count']}, Avg Confidence: {error['confidence']:.3f}")
    
    # Save all results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'evaluation_results/metrics_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nAll evaluation results have been saved to the 'evaluation_results' directory")
    
    return metrics

def predict_image(model, image_path, class_names):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    return class_names[predicted_class], confidence

def load_data():
    # Load metadata
    skin_df = pd.read_csv('archive/HAM10000_metadata.csv')
    le = LabelEncoder()
    skin_df['label'] = le.fit_transform(skin_df['dx'])
    class_names = le.classes_  # Store class names for later use
    
    # Get image paths
    image_path = {os.path.splitext(os.path.basename(x))[0]: x 
                 for x in glob(os.path.join('archive/', '*', '*.jpg'))}
    skin_df['path'] = skin_df['image_id'].map(image_path.get)
    
    # Load and preprocess images
    images = []
    for path in skin_df['path']:
        img = load_and_preprocess_image(path)
        images.append(img)
    
    X = np.array(images)
    Y = to_categorical(skin_df['label'], num_classes=NUM_CLASSES)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    
    return x_train, x_test, y_train, y_test, class_names

if __name__ == "__main__":
    # Load data
    x_train, x_test, y_train, y_test, class_names = load_data()
    

    model = create_model()
    print("Number of layers:", len(model.layers))
    model.summary()
    
    
    train_generator, val_generator = create_data_generators(x_train, x_test, y_train, y_test)
    
    history = train_model(model, train_generator, val_generator)
    
    evaluate_model(model, x_test, y_test, class_names) 