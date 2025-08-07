import torch
import numpy as np
import os
import argparse
from src.model import CrossModalTransformer
from src.dataset import get_logo_dataloaders
from src.utils import set_seed

def train_model(model, train_loader, val_loader, device, n_epochs=50, lr=1e-4):
    """Training function with adversarial domain adaptation"""
    model.to(device)

    # Optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Loss functions
    emotion_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Gradient reversal parameter (increases over time)
        alpha = 2.0 / (1.0 + np.exp(-10 * epoch / n_epochs)) - 1.0

        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            eye = batch['eye'].to(device)
            eeg_mask = batch['eeg_mask'].to(device)
            eye_mask = batch['eye_mask'].to(device)
            labels = batch['label'].to(device)
            groups = batch['group'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(eeg, eye, eeg_mask, eye_mask, groups, alpha)

            # Losses
            emotion_loss = emotion_criterion(outputs['emotion_logits'], labels)
            domain_loss = domain_criterion(outputs['domain_logits'], groups)

            # Total loss (emotion - domain for adversarial training)
            total_loss = emotion_loss + 0.1 * domain_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = torch.max(outputs['emotion_logits'], 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                eye = batch['eye'].to(device)
                eeg_mask = batch['eeg_mask'].to(device)
                eye_mask = batch['eye_mask'].to(device)
                labels = batch['label'].to(device)
                groups = batch['group'].to(device)

                outputs = model(eeg, eye, eeg_mask, eye_mask, groups, alpha=0.0)

                emotion_loss = emotion_criterion(outputs['emotion_logits'], labels)
                val_loss += emotion_loss.item()

                _, predicted = torch.max(outputs['emotion_logits'], 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        print(f'Epoch [{epoch+1}/{n_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Alpha: {alpha:.3f}')
        print('-' * 50)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return model

def main():
    parser = argparse.ArgumentParser(description='Train Cross-Modal Transformer for Emotion Recognition')
    parser.add_argument('--data_dir', type=str, default='SEED-V', help='Path to SEED-V dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model parameters
    model_params = {
        'eeg_features': 310,
        'eye_features': 33,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_encoder_layers': args.n_layers,
        'n_classes': 5,
        'n_subjects': 16,
        'dropout': args.dropout
    }

    # LOGO Cross-validation
    fold_results = []

    for train_loader, val_loader, test_loader, fold_info in get_logo_dataloaders(args.data_dir, args.batch_size):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_info['fold']}: Testing on Subject {fold_info['test_subject']}")
        print(f"{'='*60}")

        # Initialize model
        model = CrossModalTransformer(**model_params)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Train model
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)

        # Test evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                eeg = batch['eeg'].to(device)
                eye = batch['eye'].to(device)
                eeg_mask = batch['eeg_mask'].to(device)
                eye_mask = batch['eye_mask'].to(device)
                labels = batch['label'].to(device)
                groups = batch['group'].to(device)

                outputs = model(eeg, eye, eeg_mask, eye_mask, groups, alpha=0.0)

                _, predicted = torch.max(outputs['emotion_logits'], 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100.0 * test_correct / test_total
        fold_results.append(test_acc)

        print(f"\nFold {fold_info['fold']} Test Accuracy: {test_acc:.2f}%")

        # Save fold-specific model
        torch.save(model.state_dict(), 
                   os.path.join(args.save_dir, f'model_fold_{fold_info["fold"]}.pth'))

    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Per-fold accuracies: {[f'{acc:.2f}%' for acc in fold_results]}")
    print(f"Mean accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%")
    print(f"Best fold: {max(fold_results):.2f}%")
    print(f"Worst fold: {min(fold_results):.2f}%")

if __name__ == "__main__":
    main()
