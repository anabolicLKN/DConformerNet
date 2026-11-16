import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
import time
import torch.cuda

# Note: Ensure that functional.reset_net(model) is defined in your code.

def train_single(epochs, batch_size, model, loss_fn, dataset, learning_rate, model_name,set_device):
    """
    Trains the given model using 5-fold cross-validation.
    For each fold, the model is trained on 80% of the data and validated on 20%.
    
    Parameters
    ----------
    epochs : int
        Number of epochs to train for each fold.
    batch_size : int
        Batch size for training.
    model : torch.nn.Module
        The model to be trained.
    loss_fn : torch.nn.Module
        The loss function.
    dataset : torch.utils.data.Dataset
        The full dataset on which to perform cross-validation.
    learning_rate : float
        The initial learning rate.
    model_name : str
        Name identifier for saving models.
    
    Returns
    -------
    None
    """
    device = set_device
    
    # 모델 이름으로 결과 저장할 디렉토리 생성
    results_dir = f"results/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 모델 가중치 저장할 디렉토리 생성
    models_dir = f"{results_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 혼동 행렬 저장할 디렉토리 생성
    cm_dir = f"{results_dir}/confusion_matrices"
    os.makedirs(cm_dir, exist_ok=True)
    
    # 학습 그래프 저장할 디렉토리 생성
    plots_dir = f"{results_dir}/training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 결과 저장을 위한 추가 디렉토리
    inference_dir = f"{results_dir}/inference_time"
    os.makedirs(inference_dir, exist_ok=True)
    
    # KFold를 StratifiedKFold로 변경 (클래스 균형 유지)
    # 데이터셋에서 레이블 추출
    all_labels = []
    for _, label in dataset:
        all_labels.append(label.item())
    all_labels = np.array(all_labels)
    
    # StratifiedKFold 사용
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    fold_index = 0
    
    # For storing metrics across all folds
    all_f1_scores = []
    
    # 인퍼런스 타임 결과 저장용 리스트
    inference_results = []
    
    for train_idx, val_idx in skf.split(np.arange(len(dataset)), all_labels):
        fold_index += 1
        print("=====================================================")
        print(f"Fold {fold_index} training started.")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Reset model weights for the current fold
        current_model = copy.deepcopy(model)
        current_model = current_model.to(device)
        current_loss_fn = loss_fn.to(device)
        optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15, verbose=True)
        
        best_val_acc = 0.0
        best_model_path = f"{models_dir}/{model_name}_fold{fold_index}_best_model.pth"
        
        # Store training history for plotting (optional)
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        
        for epoch in range(epochs):
            # Training phase
            current_model.train()
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0
            for data, target in train_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                output = current_model(data)
                loss = current_loss_fn(output, target)
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, dim=1)
                epoch_train_correct += (predicted == target).sum().item()
                epoch_train_total += data.size(0)
            
            train_loss_epoch = epoch_train_loss / epoch_train_total
            train_acc_epoch = epoch_train_correct / epoch_train_total
            train_loss_list.append(train_loss_epoch)
            train_acc_list.append(train_acc_epoch)
            
            # Validation phase
            current_model.eval()
            epoch_val_loss = 0.0
            epoch_val_correct = 0
            epoch_val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    output = current_model(data)
                    loss_val = current_loss_fn(output, target)
                    functional.reset_net(current_model)
                    epoch_val_loss += loss_val.item() * data.size(0)
                    _, pred = torch.max(output, dim=1)
                    epoch_val_correct += (pred == target).sum().item()
                    epoch_val_total += data.size(0)
            epoch_val_loss /= epoch_val_total
            epoch_val_acc = epoch_val_correct / epoch_val_total
            val_loss_list.append(epoch_val_loss)
            val_acc_list.append(epoch_val_acc)
            
            print(f"Fold {fold_index}, Epoch {epoch}: "
                  f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch*100:.2f}%, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
            
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(current_model.state_dict(), best_model_path)
                print(f"Fold {fold_index}, Epoch {epoch}: Best model saved with Val Acc: {epoch_val_acc*100:.2f}%")
            
            scheduler.step(epoch_val_acc)
        
        # Load best model for current fold and evaluate final validation metrics
        current_model.load_state_dict(torch.load(best_model_path))
        current_model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        
        # For confusion matrix and F1 score
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = current_model(data)
                loss_val = current_loss_fn(output, target)
                final_val_loss += loss_val.item() * data.size(0)
                _, pred = torch.max(output, dim=1)
                final_val_correct += (pred == target).sum().item()
                final_val_total += data.size(0)
                
                # Store predictions and targets for metrics calculation
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        final_val_loss /= final_val_total
        final_val_acc = final_val_correct / final_val_total
        
        # Calculate F1 score (macro averaging)
        f1 = f1_score(all_targets, all_preds, average='macro')
        all_f1_scores.append(f1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Save F1 score to results
        fold_results.append({
            'fold': fold_index,
            'val_loss': final_val_loss,
            'val_acc': final_val_acc,
            'f1_score': f1
        })
        
        print(f"Fold {fold_index} Final Val Loss: {final_val_loss:.4f}, Final Val Acc: {final_val_acc*100:.2f}%, F1 Score: {f1:.4f}")
        
        # 모델 전체 저장 (가중치뿐만 아니라 구조도 함께 저장)
        full_model_path = f"{models_dir}/{model_name}_fold{fold_index}_full_model.pt"
        torch.save(current_model, full_model_path)
        
        # Save confusion matrix as figure
        plt.figure(figsize=(12, 10))
        
        # 글꼴 설정 증가
        plt.rcParams.update({
            'font.size': 14,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold'
        })
        
        # 혼동 행렬 그리기 (글꼴 크기와 굵기 조정)
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     annot_kws={"size": 16, "weight": "bold"},
                     cbar_kws={"shrink": 0.8})
        
        # 제목과 라벨 설정 (굵게)
        plt.title(f'Confusion Matrix - Fold {fold_index}', fontsize=18, fontweight='bold')
        plt.ylabel('True Label', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
        
        # 틱 라벨 설정
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        
        # 레이아웃 조정하여 모든 내용이 보이도록
        plt.tight_layout()
        
        cm_path = f"{cm_dir}/{model_name}_fold{fold_index}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)  # DPI 높여서 선명하게 저장
        plt.close()
        
        # 설정 원래대로 복원
        plt.rcParams.update({
            'font.size': plt.rcParamsDefault['font.size'],
            'font.weight': plt.rcParamsDefault['font.weight'],
            'axes.labelweight': plt.rcParamsDefault['axes.labelweight'],
            'axes.titleweight': plt.rcParamsDefault['axes.titleweight']
        })
        
        # (Optional) Save confusion matrix as CSV
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(f"{cm_dir}/{model_name}_fold{fold_index}_confusion_matrix.csv")
        
        # (Optional) Plot training trends for the current fold
        epochs_range = range(epochs)
        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(epochs_range, train_loss_list, label="Train Loss")
        plt.plot(epochs_range, val_loss_list, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold_index} Loss Trend")
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(epochs_range, [acc*100 for acc in train_acc_list], label="Train Accuracy")
        plt.plot(epochs_range, [acc*100 for acc in val_acc_list], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Fold {fold_index} Accuracy Trend")
        plt.legend()
        plt.tight_layout()
        train_plot_path = f"{plots_dir}/{model_name}_fold{fold_index}_training_plot.png"
        plt.savefig(train_plot_path)
        plt.close()
        
        # 모델 학습 완료 후, 인퍼런스 타임 측정
        print(f"Fold {fold_index} - Measuring inference time...")
        current_model.eval()  # 평가 모드 설정
        
        # 배치 인퍼런스 타임 측정
        batch_inference_times = []
        sample_inference_times = []
        
        with torch.no_grad():
            # 배치 단위 측정
            for data, _ in val_loader:
                data = data.to(device, non_blocking=True)
                
                # GPU 연산 동기화
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                start_time = time.time()
                _ = current_model(data)
                
                # GPU 연산 동기화
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                end_time = time.time()
                batch_time = end_time - start_time
                batch_inference_times.append(batch_time)
            
            # 단일 샘플 단위 측정 (10회 반복)
            single_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            for i, (data, _) in enumerate(single_loader):
                if i >= 10:  # 첫 10개 샘플만 측정
                    break
                
                data = data.to(device, non_blocking=True)
                
                # GPU 연산 동기화
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                start_time = time.time()
                _ = current_model(data)
                functional.reset_net(current_model)
                
                # GPU 연산 동기화
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                end_time = time.time()
                sample_time = end_time - start_time
                sample_inference_times.append(sample_time)
        
        # 평균 인퍼런스 타임 계산
        avg_batch_time = np.mean(batch_inference_times)
        avg_batch_std = np.std(batch_inference_times)
        avg_sample_time = np.mean(sample_inference_times)
        avg_sample_std = np.std(sample_inference_times)
        
        # 배치당 샘플 수
        samples_per_batch = batch_size
        
        # 인퍼런스 결과 저장
        inference_result = {
            'fold': fold_index,
            'avg_batch_time': avg_batch_time,
            'avg_batch_std': avg_batch_std,
            'avg_sample_time': avg_sample_time,
            'avg_sample_std': avg_sample_std,
            'batch_size': samples_per_batch,
            'samples_per_second_batch': samples_per_batch / avg_batch_time,
            'samples_per_second_single': 1.0 / avg_sample_time
        }
        inference_results.append(inference_result)
        
        print(f"Fold {fold_index} Inference Time:")
        print(f"  Batch mode ({samples_per_batch} samples): {avg_batch_time*1000:.2f} ms ± {avg_batch_std*1000:.2f} ms")
        print(f"  Single sample mode: {avg_sample_time*1000:.2f} ms ± {avg_sample_std*1000:.2f} ms")
        print(f"  Throughput (batch): {samples_per_batch/avg_batch_time:.2f} samples/sec")
        print(f"  Throughput (single): {1.0/avg_sample_time:.2f} samples/sec")
    
    # Calculate average metrics over all folds
    avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
    avg_val_acc = np.mean([result['val_acc'] for result in fold_results])
    avg_f1 = np.mean(all_f1_scores)
    
    print("=====================================================")
    # 각 폴드별 Accuracy, F1 Score 출력
    for result in fold_results:
        print(f"Fold {result['fold']} - Accuracy: {result['val_acc']*100:.2f}%, F1 Score: {result['f1_score']:.4f}")
    
    # 평균 Loss, Accuracy, F1 Score 출력
    print(f"5-Fold Cross Validation Results:")
    print(f"Average Val Loss: {avg_val_loss:.4f}")
    print(f"Average Val Acc: {avg_val_acc*100:.2f}%")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Save overall results to CSV
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{results_dir}/{model_name}_cv_results.csv", index=False)
    
    # 최종 평균 혼동 행렬 생성 (선택적)
    plt.figure(figsize=(12, 10))
    plt.title(f'{model_name} - Average Confusion Matrix')
    plt.text(0.5, 0.02, f'Avg Val Acc: {avg_val_acc*100:.2f}%, Avg F1 Score: {avg_f1:.4f}', 
             horizontalalignment='center', fontsize=12, transform=plt.gcf().transFigure)
    plt.savefig(f"{results_dir}/{model_name}_avg_metrics.png")
    plt.close()
    
    # Save summary metrics
    summary = {
        'model_name': model_name,
        'avg_val_loss': avg_val_loss,
        'avg_val_acc': avg_val_acc,
        'avg_f1_score': avg_f1,
        'fold_count': len(fold_results),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Append to summary file (useful for comparing multiple model runs)
    summary_df = pd.DataFrame([summary])
    summary_path = "results/model_summary.csv"
    
    if os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        updated_summary = pd.concat([existing_summary, summary_df], ignore_index=True)
        updated_summary.to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)
    
    # 모든 폴드의 결과를 요약한 README 파일 생성
    with open(f"{results_dir}/README.md", "w") as f:
        f.write(f"# {model_name} 학습 결과 요약\n\n")
        f.write(f"학습 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 5-Fold 교차 검증 평균 성능\n")
        f.write(f"- 평균 검증 손실(Val Loss): {avg_val_loss:.4f}\n")
        f.write(f"- 평균 정확도(Accuracy): {avg_val_acc*100:.2f}%\n")
        f.write(f"- 평균 F1 점수: {avg_f1:.4f}\n\n")
        f.write(f"## 폴드별 성능\n")
        for result in fold_results:
            f.write(f"### Fold {result['fold']}\n")
            f.write(f"- 검증 손실(Val Loss): {result['val_loss']:.4f}\n")
            f.write(f"- 정확도(Accuracy): {result['val_acc']*100:.2f}%\n")
            f.write(f"- F1 점수: {result['f1_score']:.4f}\n\n")
        f.write("\n\n이 README 파일은 자동으로 생성되었습니다.")
    
    # 평균 인퍼런스 타임 계산 및 출력
    avg_batch_inference = np.mean([r['avg_batch_time'] for r in inference_results])
    avg_sample_inference = np.mean([r['avg_sample_time'] for r in inference_results])
    avg_batch_throughput = np.mean([r['samples_per_second_batch'] for r in inference_results])
    avg_sample_throughput = np.mean([r['samples_per_second_single'] for r in inference_results])
    
    print("=====================================================")
    print(f"Average Inference Time Across All Folds:")
    print(f"  Batch mode ({batch_size} samples): {avg_batch_inference*1000:.2f} ms")
    print(f"  Single sample mode: {avg_sample_inference*1000:.2f} ms")
    print(f"  Throughput (batch): {avg_batch_throughput:.2f} samples/sec")
    print(f"  Throughput (single): {avg_sample_throughput:.2f} samples/sec")
    
    # 인퍼런스 결과 CSV로 저장
    inference_df = pd.DataFrame(inference_results)
    inference_df.to_csv(f"{inference_dir}/{model_name}_inference_time.csv", index=False)
    
    # README 파일에 인퍼런스 정보 추가
    with open(f"{results_dir}/README.md", "a") as f:
        f.write(f"\n## 인퍼런스 시간 측정 결과\n")
        f.write(f"- 배치 모드 ({batch_size} 샘플): {avg_batch_inference*1000:.2f} ms\n")
        f.write(f"- 단일 샘플 모드: {avg_sample_inference*1000:.2f} ms\n")
        f.write(f"- 배치 처리량: {avg_batch_throughput:.2f} 샘플/초\n")
        f.write(f"- 단일 샘플 처리량: {avg_sample_throughput:.2f} 샘플/초\n\n")

