from multivariate_cwru import CWRU
import numpy as np
import torch
import torch.utils.data as data
def simplify_labels(y, labels):
    """레이블을 4개의 클래스로 단순화합니다 (Normal: 0, Ball: 1, IR: 2, OR: 3)"""
    new_y = y.copy()
    for i, label in enumerate(labels):
        label = str(label[1]).upper()  # 튜플의 두 번째 요소를 문자열로 변환하고 대문자로
        if 'NORMAL' in label:
            new_y[y == i] = 0  # Normal
        elif 'BALL' in label:
            new_y[y == i] = 1  # Ball
        elif 'INNERRACE' in label or 'IR' in label:
            new_y[y == i] = 2  # Inner Race
        elif 'OUTERRACE' in label or 'OR' in label:
            new_y[y == i] = 3  # Outer Race
    
    # 레이블이 제대로 변환되었는지 확인
    unique_labels = np.unique(new_y)
    if not all(label in [0, 1, 2, 3] for label in unique_labels):
        raise ValueError(f"잘못된 레이블이 있습니다: {unique_labels}")
    
    return new_y

def add_noise(signal, snr_db):
    """
    신호에 지정된 SNR 레벨의 가우시안 노이즈를 추가합니다.
    
    Parameters:
    -----------
    signal : numpy.ndarray or torch.Tensor
        원본 신호
    snr_db : float
        신호 대 잡음 비율(dB)
        
    Returns:
    --------
    noisy_signal : numpy.ndarray or torch.Tensor
        노이즈가 추가된 신호
    """
    if snr_db is None:
        return signal
        
    # 텐서를 numpy로 변환
    is_tensor = isinstance(signal, torch.Tensor)
    if is_tensor:
        original_device = signal.device
        signal_np = signal.cpu().numpy()
    else:
        signal_np = signal
    
    # 신호 에너지 계산
    signal_power = np.mean(signal_np ** 2)
    
    # SNR을 선형 단위로 변환
    snr_linear = 10 ** (snr_db / 10)
    
    # 노이즈 파워 계산
    noise_power = signal_power / snr_linear
    
    # 노이즈 생성
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal_np.shape)
    
    # 신호에 노이즈 추가
    noisy_signal = signal_np + noise
    
    # 원래 형식으로 변환
    if is_tensor:
        noisy_signal = torch.tensor(noisy_signal, dtype=signal.dtype, device=original_device)
        
    return noisy_signal
def get_cwru_dataset_cross_val(experiment="12DriveEndFault", length=2048, train_split=0.8,
                               seed=1, sens=1, rpms=None, normal=True, balance_samples=None,
                               snr_db=None):
    """
    CWRU 데이터셋을 로드하고 하나의 TensorDataset으로 반환합니다.
    이 함수는 5-Fold 교차 검증을 위한 전체 데이터를 하나의 데이터셋으로 만듭니다.
    
    Parameters:
    -----------
    experiment : str
        실험 유형
    length : int
        신호 길이
    train_split : float
        학습 데이터 비율
    seed : int
        랜덤 시드
    sens : int
        센서 번호
    rpms : list
        사용할 RPM 리스트
    normal : bool
        정상 상태 데이터 포함 여부
    balance_samples : int
        각 클래스별 샘플 수
    snr_db : float, optional
        신호 대 잡음 비율(dB). None이면 노이즈 추가 안함
        
    Returns:
    --------
    dataset : TensorDataset
        전체 데이터셋 (교차 검증을 위해 하나의 데이터셋)
    """
    if rpms is None:
        rpms = ['1797', '1772', '1750', '1730']
        
    # CWRU 데이터 로드 (CWRU 함수는 별도로 정의되어 있어야 합니다.)
    data = CWRU(experiment, length, train_split, seed, sens, *rpms, 
                normal_condition=normal)
    
    # 전체 데이터 합치기 (학습 데이터와 테스트 데이터를 합침)
    X = np.concatenate((data.x_train, data.x_test), axis=0)
    y = np.concatenate((data.y_train, data.y_test), axis=0)
    
    # 원래 라벨 확인
    print("Original unique labels:", np.unique(y))
    print("Original labels:", data.labels)
    
    # 원본 레이블 분포 확인
    print("\n원본 레이블 분포:")
    unique_original, counts_original = np.unique(y, return_counts=True)
    for orig_lbl, count in zip(unique_original, counts_original):
        label_info = data.labels[orig_lbl] if orig_lbl < len(data.labels) else "Unknown"
        print(f"원본 레이블 {orig_lbl} ({label_info}): {count}개 샘플")
    
    # 라벨 단순화
    y = simplify_labels(y, data.labels)
    
    # 단순화된 라벨 확인
    print("\nSimplified unique labels:", np.unique(y))
    
    # 클래스별로 지정된 샘플 수만큼 선택
    if balance_samples is not None and balance_samples > 0:
        X_balanced = []
        y_balanced = []
        
        for cls in np.unique(y):
            # 현재 클래스의 인덱스 찾기
            indices = np.where(y == cls)[0]
            
            # 랜덤 샘플링 (지정된 개수만큼)
            if len(indices) > balance_samples:
                # 랜덤하게 balance_samples개 선택
                selected_indices = np.random.choice(indices, balance_samples, replace=False)
            else:
                # 부족하면 중복 허용하여 선택 (오버샘플링)
                selected_indices = np.random.choice(indices, balance_samples, replace=True)
                print(f"클래스 {cls}는 샘플이 부족하여 오버샘플링됨 ({len(indices)} -> {balance_samples})")
            
            # 선택된 샘플 추가
            X_balanced.append(X[selected_indices])
            y_balanced.append(y[selected_indices])
        
        # 모든 클래스의 데이터 합치기
        X = np.concatenate(X_balanced)
        y = np.concatenate(y_balanced)
        
        print(f"\n각 클래스당 {balance_samples}개 샘플로 균형화된 데이터셋 생성됨")
    else:
        print("\n균형화를 적용하지 않음 - 전체 데이터 사용")
    
    # numpy -> tensor 변환
    X_tensor = torch.FloatTensor(X)
    
    # SNR에 따라 노이즈 추가
    if snr_db is not None:
        print(f"SNR {snr_db}dB의 노이즈 추가 중...")
        X_tensor = add_noise(X_tensor, snr_db)
    
    y_tensor = torch.LongTensor(y)
    
    # 전체 데이터를 하나의 TensorDataset으로 생성
    dataset = TensorDataset(X_tensor, y_tensor)
    
    print(f"\nTotal dataset samples: {len(dataset)}")
    
    # 클래스별 샘플 수 확인
    print("\n단순화된 레이블의 클래스별 샘플 수:")
    class_counts = count_samples_per_class(dataset)
    
    # 클래스별 비율 계산 추가
    total = len(dataset)
    print("\n클래스별 비율:")
    for cls, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"클래스 {cls}: {percentage:.2f}%")
    
    return dataset

# 데이터셋에서 각 클래스별 샘플 수 확인하는 코드
def count_samples_per_class(dataset):
    """
    데이터셋의 각 클래스별 샘플 수를 계산하여 출력합니다.
    
    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        샘플 수를 확인할 데이터셋
    """
    # 데이터셋이 TensorDataset인 경우
    if isinstance(dataset, torch.utils.data.TensorDataset):
        labels = dataset.tensors[1].numpy()
    else:
        # 일반 데이터셋의 경우 모든 레이블 수집
        labels = []
        for _, label in dataset:
            if isinstance(label, torch.Tensor):
                labels.append(label.item())
            else:
                labels.append(label)
        labels = np.array(labels)
    
    # 유니크한 클래스 찾기
    unique_classes = np.unique(labels)
    
    # 각 클래스별 샘플 수 계산
    class_counts = {}
    for cls in unique_classes:
        count = np.sum(labels == cls)
        class_counts[cls] = count
        print(f"클래스 {cls}: {count}개 샘플")
    
    # 전체 샘플 수 및 요약 정보
    total_samples = len(labels)
    print(f"\n전체 샘플 수: {total_samples}")
    print(f"클래스 수: {len(unique_classes)}")
    
    # 클래스 불균형 확인
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"클래스 불균형 비율 (최대/최소): {imbalance_ratio:.2f}")
    
    return class_counts