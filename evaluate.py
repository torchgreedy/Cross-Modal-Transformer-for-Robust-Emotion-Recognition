import torch
import numpy as np
import argparse
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import CrossModalTransformer
from src.dataset import get_logo_dataloaders
from src.utils import set_seed

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data and return metrics and visualizations"""
    model.eval()
    all_preds = []
    all_labels = []
    all_eeg_importance = []
    all_eye_importance = []
    all_cross_attn = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            eye = batch['eye'].to(device)
            eeg_mask = batch['eeg_mask'].to(device)
            eye_mask = batch['eye_mask'].to(device)
            labels = batch['label'].to(device)
            groups = batch['group'].to(device)

            outputs = model(eeg, eye, eeg_mask, eye_mask, groups, alpha=0.0)
            
            _, predictions = torch.max(outputs['emotion_logits'], 1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Collect attention and importance weights for interpretability
            all_eeg_importance.append(outputs['eeg_importance'].cpu().numpy())
            all_eye_importance.append(outputs['eye_importance'].cpu().numpy())
            all_cross_attn.append(outputs['cross_modal_attn'])
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels) * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, 
                                        target_names=["Happy", "Sad", "Neutral", "Fear", "Disgust"])
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': all_preds,
        'true_labels': all_labels,
        'eeg_importance': all_eeg_importance,
        'eye_importance': all_eye_importance,
        'cross_attention': all_cross_attn
    }

def visualize_results(results, save_dir):
    """Create and save visualizations of model performance"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results['confusion_matrix'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=["Happy", "Sad", "Neutral", "Fear", "Disgust"],
        yticklabels=["Happy", "Sad", "Neutral", "Fear", "Disgust"]
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Save classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write(results['classification_report'])

def main():
    parser = argparse.ArgumentParser(description='Evaluate Cross-Modal Transformer for Emotion Recognition')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='SEED-V', help='Path to SEED-V dataset')
    parser.add_argument('--subject_id', type=int, default=None, help='Subject ID to evaluate (1-16)')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters
    model_params = {
        'eeg_features': 310,
        'eye_features': 33,
        'd_model': 512,
        'n_heads': 8,
        'n_encoder_layers': 4,
        'n_classes': 5,
        'n_subjects': 16,
        'dropout': 0.1
    }

    # Initialize model
    model = CrossModalTransformer(**model_params)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Get dataloaders for the specified subject
    target_fold = args.subject_id if args.subject_id else None
    
    for train_loader, val_loader, test_loader, fold_info in get_logo_dataloaders(args.data_dir, batch_size=32):
        # Skip folds that don't match the requested subject
        if target_fold and fold_info['test_subject'] != target_fold:
            continue
            
        print(f"\nEvaluating on Subject {fold_info['test_subject']}")
        
        # Evaluate the model
        results = evaluate_model(model, test_loader, device)
        
        # Save and visualize results
        subject_save_dir = os.path.join(args.save_dir, f"subject_{fold_info['test_subject']}")
        visualize_results(results, subject_save_dir)
        
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Results saved to {subject_save_dir}")
        
        # If a specific subject was requested, exit after processing
        if target_fold:
            break

if __name__ == "__main__":
    main()
