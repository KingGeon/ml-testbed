import os
import torch
from src.models.components.conv_lstm_classifier_no_dropout_outputsize_control import CONV_LSTM_Classifier
from typing import List,Tuple
import librosa
import numpy as np
import pandas as pd
import math
import scipy
import random
import tqdm
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_score


fault_type_dict = {"정상": 0,
                "베어링불량": 1,
                "벨트느슨함": 2,
                "축정렬불량": 3,
                "회전체불평형": 4}
root: str = "/home/mongoose01/mongooseai/data/cms/open_source/AI_hub/기계시설물 고장 예지 센서/Training/vibration"

def sampling_with_bandpass(data , sf, rf):
    rotational_frequency = rf  # 회전 주파수 30Hz
    frequencies_to_filter = [rotational_frequency * i for i in range(1, 9)]  # 1배, 2배, 3배, 4배 주파수
    frequencies_to_filter.append(120)  # 추가적으로 120Hz 필터링

    # 대역폭 설정 (+/- 10Hz)
    bandwidth = 10

    # 예제 신호 생성: 여러 주파수 성분과 잡음 포함

    # 필터링된 신호를 저장할 딕셔너리
    filtered_signals = []

    # 각 주파수 대역에 대한 필터링
    for f in frequencies_to_filter:
        band = [f - bandwidth, f + bandwidth]
        b, a = signal.butter(4, [band[0]/(sf/2), band[1]/(sf/2)], btype='band')
        analytic_signal = signal.hilbert(signal.filtfilt(b, a, data))
        # Compute the envelope
        envelope = np.abs(analytic_signal)
        filtered_signals.append(envelope)

    return np.stack(filtered_signals, axis=1)

def original_fft(signal,sf):
    _signal = signal
    y = np.abs(np.fft.fft(_signal,axis = 0)/len(signal))        # fft computing and normaliation
    y = y[range(math.trunc(len(signal)/2))]  
    k = np.arange(len(signal))
    f0=k*sf/len(signal)    # double sides frequency range
    f0=f0[range(math.trunc(len(signal)/2))]  
    return f0, y

def is_peak_at_frequency(freq, spectrum,threshold):
    # 실제 구현에선 주파수 스펙트럼에서 해당 주파수의 진폭이 피크를 형성하는지 검사
    return spectrum[int(freq)] > threshold
                
def estimate_rpm(numpy_array, sf=8192, f_min=27.6, f_max=29.1, f_r=1, M=60, c=2):

    f, magnitude_spectrum = original_fft(numpy_array, sf)
    # 속도 후보들과 그에 대한 초기 확률 설정
    candidates = np.arange(f_min, f_max, f_r/M)
    probabilities = np.ones_like(candidates)  # 모든 후보에 동일한 초기 확률 할당
    #print(f"3분위수 :{np.percentile(magnitude_spectrum, 75)}")
    #print(f"중위수 : {np.percentile(magnitude_spectrum, 50)}")
    #print(f"평균 : {np.mean(magnitude_spectrum)}")
    #print(f"최대 : {np.max(magnitude_spectrum)}")
    threshold = np.mean(magnitude_spectrum)*1.5
    # 후보들의 조화 검사 및 확률 업데이트
    for i, fc in enumerate(candidates):
        for k in range(1, M+1):
            harmonic_freq = k * fc
                    
            if not is_peak_at_frequency(harmonic_freq, magnitude_spectrum,threshold):
                probabilities[i] /= c  # 피크가 없으면 확률 감소
                # 최종 추정 속도 결정
    estimated_speed = candidates[np.argmax(probabilities)]


    #print(f'Estimated speed: {estimated_speed} Estimated rpm: {estimated_speed*60}')
    return estimated_speed*60

def engine_order_fft(signal, rpm, sf = 8192, res = 100, ts = 1):
    _signal = signal[:int(sf*ts)]
    pad_length = int(sf*(res * 60/rpm - ts))
    zero_padded_signal = np.concatenate((_signal, np.zeros(pad_length)))
    y = np.abs(np.fft.fft(zero_padded_signal)/len(zero_padded_signal))     
    y = y[range(math.trunc(len(signal)))]     
    return y[:sf]

def min_max_scaling(data, new_min=-1, new_max=1):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = new_min + (data - min_val) * (new_max - new_min) / (max_val - min_val)
    return scaled_data

def standard_scaling(data):
    mean = np.mean(data,axis = 0 )
    std = np.std(data,axis = 0 )
    scaled_data = (data - mean) / (std + 1e-8)
    return scaled_data
import scipy
import random
def envelope_spectrum(signal, fs):
        """
        Compute the envelope spectrum of a signal.
        
        Args:
        signal (numpy array): Input signal.
        fs (float): Sampling frequency of the signal.
        
        Returns:
        freqs (numpy array): Frequencies of the envelope spectrum.
        envelope (numpy array): Envelope spectrum of the signal.
        """
        # Compute the analytic signal using Hilbert transform
        analytic_signal = scipy.signal.hilbert(signal)
        # Compute the envelope
        envelope = np.abs(analytic_signal)

        # Compute the FFT of the envelope
        envelope_fft = np.fft.fft(envelope)
        # Compute the corresponding frequencies
        freqs = np.fft.fftfreq(len(envelope), 1 / fs)
        
        # Take only the positive frequencies and their corresponding FFT values

        freqs = freqs[:]
        envelope_fft = np.abs(envelope_fft[:])

        return envelope, envelope_fft
def up_sample(input_wav, origin_sr, resample_sr, upsample_method):
        resample = librosa.resample(y=input_wav, orig_sr=origin_sr, target_sr=resample_sr, res_type=upsample_method)
        return resample

def calculate_rms(input):
    input = input.reshape(-1,1)#make input 1D array
    """
    input : signal, 1D array

    Do: calculate rms for input 
    """
    return np.sqrt(np.mean(input**2))

 


def process_csv( csv_path, fault):
        data = np.loadtxt(csv_path, delimiter=',', skiprows=9, usecols=(1,), max_rows=4000)
        data = standard_scaling(data*100)
        data = up_sample(data, 4000, 8192, "soxr_vhq")
        rpm = estimate_rpm(data)
        bandpassed_signal = sampling_with_bandpass(data, 8192, rpm/60)
        eofft = engine_order_fft(data,rpm)
        envelope, envelop_spectrum = envelope_spectrum(data,8192)

        return data.reshape(-1,1), bandpassed_signal.reshape(-1,9), envelope.reshape(-1,1), envelop_spectrum.reshape(-1,1), eofft.reshape(-1,1), fault_type_dict[fault]
    
    # Process the CSV files for each motor power
def process_motor_power( data_root, motor_power, jump_size, fault_type_count_list,motor_dict):
        data_list = []
        target_list = []
        motor_power_path = os.path.join(data_root, motor_power)
        for motor in os.listdir(motor_power_path):
            for fault in os.listdir(os.path.join(motor_power_path, motor)):
                if fault == "정상":
                      continue
                motor_path = os.path.join(motor_power_path, motor, fault)
                csv_list = [file for file in os.listdir(motor_path) if file.endswith('.csv')]
                random.shuffle(csv_list)    
                cnt = 0
                for csv in csv_list[0:int(len(csv_list)):jump_size]:
                    
                    numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, target = process_csv(os.path.join(motor_path,csv), fault)
                    motor_power_array = np.ones_like(numpy_array)*float(str(motor_power[:-2]))
                    motor_array = np.ones_like(numpy_array)*motor_dict[motor]
                    """
                    np.save(os.path.join(motor_path, 'numpy_array_' + str(cnt)+ '.npy'), numpy_array)
                    np.save(os.path.join(motor_path, 'bandpassed_signal_' + str(cnt)+ '.npy'), bandpassed_signal)
                    np.save(os.path.join(motor_path, 'envelope_' + str(cnt)+ '.npy'), envelope)
                    np.save(os.path.join(motor_path, 'envelop_spectrum_' + str(cnt)+ '.npy'), envelop_spectrum)
                    np.save(os.path.join(motor_path, 'eofft_' + str(cnt)+ '.npy'), eofft)
                    np.save(os.path.join(motor_path, 'target_' + str(cnt)+ '.npy'), target)
                    """
                    cnt += 1
                    data = np.concatenate([numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, motor_power_array,motor_array], axis = -1)
                    data_list.append(data)
                    target_list.append(target)
                    
                    """
                    numpy_array = np.load(os.path.join(motor_path, 'numpy_array_' + str(cnt)+ '.npy'))
                    motor_power_array = np.ones_like(numpy_array)*float(str(motor_power[:-2]))
                    bandpassed_signal_array = np.load(os.path.join(motor_path, 'bandpassed_signal_' + str(cnt)+ '.npy'))
                    envelope_array = np.load(os.path.join(motor_path, 'envelope_' + str(cnt)+ '.npy'))
                    envelop_spectrum_array = np.load(os.path.join(motor_path, 'envelop_spectrum_' + str(cnt)+ '.npy'))
                    eofft = np.load(os.path.join(motor_path, 'eofft_' + str(cnt)+ '.npy'))
                    target = np.load(os.path.join(motor_path, 'target_' + str(cnt)+ '.npy'))
                    cnt += 1
                    data = np.concatenate([numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, motor_power_array], axis = -1)
                    data_list.append(data)
                    target_list.append(target.item())
                    """
        return data_list, target_list
def process_motor_power_normal( data_root, motor_power, jump_size, fault_type_count_list,motor_dict):
        data_list = []
        target_list = []
        motor_power_path = os.path.join(data_root, motor_power)
        for motor in os.listdir(motor_power_path):
            for fault in os.listdir(os.path.join(motor_power_path, motor)):
                if fault == "정상":
                    motor_path = os.path.join(motor_power_path, motor, fault)
                    csv_list = [file for file in os.listdir(motor_path) if file.endswith('.csv')]
                    random.shuffle(csv_list)    
                    cnt = 0
                    for csv in csv_list[0:int(len(csv_list)):jump_size]:  # Taking first 2000 files after shuffling
                        
                        numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, target = process_csv(os.path.join(motor_path,csv), fault)
                        motor_power_array = np.ones_like(numpy_array)*float(str(motor_power[:-2]))
                        motor_array = np.ones_like(numpy_array)*motor_dict[motor]
                        """
                        np.save(os.path.join(motor_path, 'numpy_array_' + str(cnt)+ '.npy'), numpy_array)
                        np.save(os.path.join(motor_path, 'bandpassed_signal_' + str(cnt)+ '.npy'), bandpassed_signal)
                        np.save(os.path.join(motor_path, 'envelope_' + str(cnt)+ '.npy'), envelope)
                        np.save(os.path.join(motor_path, 'envelop_spectrum_' + str(cnt)+ '.npy'), envelop_spectrum)
                        np.save(os.path.join(motor_path, 'eofft_' + str(cnt)+ '.npy'), eofft)
                        np.save(os.path.join(motor_path, 'target_' + str(cnt)+ '.npy'), target)
                        """
                        cnt += 1
                        data = np.concatenate([numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, motor_power_array,motor_array], axis = -1)
                        data_list.append(data)
                        target_list.append(target)
                        
                        """
                        numpy_array = np.load(os.path.join(motor_path, 'numpy_array_' + str(cnt)+ '.npy'))
                        motor_power_array = np.ones_like(numpy_array)*float(str(motor_power[:-2]))
                        bandpassed_signal_array = np.load(os.path.join(motor_path, 'bandpassed_signal_' + str(cnt)+ '.npy'))
                        envelope_array = np.load(os.path.join(motor_path, 'envelope_' + str(cnt)+ '.npy'))
                        envelop_spectrum_array = np.load(os.path.join(motor_path, 'envelop_spectrum_' + str(cnt)+ '.npy'))
                        eofft = np.load(os.path.join(motor_path, 'eofft_' + str(cnt)+ '.npy'))
                        target = np.load(os.path.join(motor_path, 'target_' + str(cnt)+ '.npy'))
                        cnt += 1
                        data = np.concatenate([numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, motor_power_array], axis = -1)
                        data_list.append(data)
                        target_list.append(target.item())
                        """
        return data_list, target_list
def process_motor_power_fault( data_root, motor_power, csv_num_to_use, fault_type_count_list,motor_dict):
        data_list = []
        target_list = []
        motor_power_path = os.path.join(data_root, motor_power)
        for motor in os.listdir(motor_power_path):
            for fault in os.listdir(os.path.join(motor_power_path, motor)):
                if fault == "정상":
                    continue
                motor_path = os.path.join(motor_power_path, motor, fault)
                csv_list = [file for file in os.listdir(motor_path) if file.endswith('.csv')]
                random.shuffle(csv_list)    
                cnt = 0
                for csv in csv_list[:int(fault_type_count_list[0]/fault_type_count_list[fault_type_dict[fault]]*csv_num_to_use)]:  # Taking first 2000 files after shuffling
                    
                    numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, target = process_csv(os.path.join(motor_path,csv), fault)
                    motor_power_array = np.ones_like(numpy_array)*float(str(motor_power[:-2]))
                    motor_array = np.ones_like(numpy_array)*motor_dict[motor]
                    """
                    np.save(os.path.join(motor_path, 'numpy_array_' + str(cnt)+ '.npy'), numpy_array)
                    np.save(os.path.join(motor_path, 'bandpassed_signal_' + str(cnt)+ '.npy'), bandpassed_signal)
                    np.save(os.path.join(motor_path, 'envelope_' + str(cnt)+ '.npy'), envelope)
                    np.save(os.path.join(motor_path, 'envelop_spectrum_' + str(cnt)+ '.npy'), envelop_spectrum)
                    np.save(os.path.join(motor_path, 'eofft_' + str(cnt)+ '.npy'), eofft)
                    np.save(os.path.join(motor_path, 'target_' + str(cnt)+ '.npy'), target)
                    """
                    cnt += 1
                    data = np.concatenate([numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, motor_power_array,motor_array], axis = -1)
                    data_list.append(data)
                    target_list.append(target)
                    
                    """
                    numpy_array = np.load(os.path.join(motor_path, 'numpy_array_' + str(cnt)+ '.npy'))
                    motor_power_array = np.ones_like(numpy_array)*float(str(motor_power[:-2]))
                    bandpassed_signal_array = np.load(os.path.join(motor_path, 'bandpassed_signal_' + str(cnt)+ '.npy'))
                    envelope_array = np.load(os.path.join(motor_path, 'envelope_' + str(cnt)+ '.npy'))
                    envelop_spectrum_array = np.load(os.path.join(motor_path, 'envelop_spectrum_' + str(cnt)+ '.npy'))
                    eofft = np.load(os.path.join(motor_path, 'eofft_' + str(cnt)+ '.npy'))
                    target = np.load(os.path.join(motor_path, 'target_' + str(cnt)+ '.npy'))
                    cnt += 1
                    data = np.concatenate([numpy_array, bandpassed_signal_array, envelope_array, envelop_spectrum_array, eofft, motor_power_array], axis = -1)
                    data_list.append(data)
                    target_list.append(target.item())
                    """
        return data_list, target_list 
def classify_with_cosine_similarity(test_data, class_means):
    # 코사인 유사도 계산
        similarities = cosine_similarity(test_data, class_means)

        # 가장 유사도가 높은 클래스를 예측 값으로 선택
        predictions = np.argmax(similarities, axis=1) + 1

        return predictions   
def classify_with_manhattan_distance(test_data, class_means):
        # 맨해튼 거리 계산
        distances = cdist(test_data, class_means, metric='cityblock')

        # 가장 거리가 짧은 클래스를 예측 값으로 선택
        predictions = np.argmin(distances, axis=1) + 1

        return predictions
def fscore():
    new_directory = "/home/geon/dev_geon/ml-testbed"  # 변경하고자 하는 새 디렉토리 경로로 대체
    os.chdir(new_directory)
    current_directory = os.getcwd()
    print("현재 작업 디렉토리:", current_directory)

    loaded_model = CONV_LSTM_Classifier(use_fft_stat = False, topk_freq=5, output_size= 128,lstm_hidden_size = 64, dense1_out_size = 64, dense2_out_size= 32, metadata_embedding = True,use_raw = False, use_envelope = True, use_stat = True,  use_eofft= True, use_envelope_spectrum = True, use_bandpass = True)

    # 모델 가중치 불러오기
    loaded_model.load_state_dict(torch.load('/home/geon/dev_geon/ml-testbed/src/models/components/ProtoNet_mixup_triplet_no_embedding.pth'))
    #ProtoNet_mixup_triplet_no_embedding
    #ProtoNet_no_mixup_triplet_no_embedding


    #["2.2kW","3.7kW","15kW","37kW","7.5kW","18.5kW","30kW","55kW","22kW"]
    #["7.5kW","18.5kW","30kW","55kW","22kW"]
    val_motor_power: List[str] = []
    test_motor_power: List[str] = ["5.5kW"]
    fault_type_dict = {
        "정상": 0,
        "베어링불량": 1,
        "벨트느슨함": 2,
        "축정렬불량": 3,
        "회전체불평형": 4
    }
    root: str = "/home/mongoose01/mongooseai/data/cms/open_source/AI_hub/기계시설물 고장 예지 센서/Training/vibration"

    motor_list = []
    train_target_list = []
    fault_type_count_list = [0,0,0,0,0]
    train_motor_power = sorted(list(set(os.listdir(root)) - set(test_motor_power) - set(val_motor_power)))#- set(val_motor_power)
    for motor_power in test_motor_power:
        motor_power_path = os.path.join(root, motor_power)
        for motor in os.listdir(motor_power_path):
            motor_list.append(motor)
    for motor_power in val_motor_power:
        motor_power_path = os.path.join(root, motor_power)
        for motor in os.listdir(motor_power_path):
            motor_list.append(motor)
    for motor_power in train_motor_power:
        motor_power_path = os.path.join(root, motor_power)  
        for motor in os.listdir(motor_power_path):
            motor_list.append(motor)
    unique_motor_list = list(set(motor_list))

    # Create a dictionary to hold motor names and their vectorized representations
    motor_dict = {}
    for index, motor_name in enumerate(unique_motor_list):
        motor_dict[motor_name] = index  
    test_data_list = []
    test_target_list = []
    test_rms_list = []
    fault_type_count_list = [0, 0, 0, 0, 0]

    for motor_power in test_motor_power:
        motor_power_path = os.path.join(root, motor_power)
        for motor in os.listdir(motor_power_path):
            for fault in os.listdir(os.path.join(motor_power_path, motor)):
                fault_type_count_list[fault_type_dict[fault]] += 1

    print(fault_type_count_list)

    for motor_power in tqdm.tqdm(test_motor_power, desc='Processing Motor Powers'):
        test_motor_data, test_motor_targets = process_motor_power(root, motor_power, 50, fault_type_count_list,motor_dict)

    fault_type_count_list = [0, 0, 0, 0, 0]
    train_motor_power = sorted(list(set(os.listdir(root))  - set(test_motor_power) - set(val_motor_power)))
    for motor_power in train_motor_power:
        motor_power_path = os.path.join(root, motor_power)
        for motor in os.listdir(motor_power_path):
            for fault in os.listdir(os.path.join(motor_power_path, motor)):
                fault_type_count_list[fault_type_dict[fault]] += 1

    print(fault_type_count_list)
    val_data_list = []
    val_target_list = []
    for motor_power in tqdm.tqdm(train_motor_power, desc='Processing Motor Powers'):
        val_motor_data, val_motor_targets = process_motor_power(root, motor_power, 200, fault_type_count_list,motor_dict)
        val_data_list.extend(val_motor_data)
        val_target_list.extend(val_motor_targets)
    """
    for motor_power in tqdm.tqdm(test_motor_power, desc='Processing Motor Powers'):
                motor_data, motor_targets = process_motor_power_normal(root, motor_power, 200, fault_type_count_list,motor_dict)
                val_data_list.extend(motor_data)
                val_target_list.extend(motor_targets)
    
    """


    test_stacked_array = np.stack(test_motor_data[:],axis = 0)
    val_stacked_array = np.stack(val_data_list[:],axis = 0)


    test_hi = loaded_model.forward(torch.Tensor(test_stacked_array))
    val_hi = loaded_model.forward(torch.Tensor(val_stacked_array))


    test_bye = loaded_model.fault_classfier(test_hi)
    val_bye = loaded_model.fault_classfier(val_hi)



    # StandardScaler를 초기화합니다.
    val_embeddings =val_bye.detach().numpy()  # requires_grad=False로 설정한 데이터
    val_target_labels = val_target_list  # 타겟 레이블 리스트

    test_embeddings =test_bye.detach().numpy()  # requires_grad=False로 설정한 데이터
    test_target_labels = test_motor_targets  # 타겟 레이블 리스트
    # X 데이터를 스케일링합니다.

   
    # test_embeddings도 동일한 평균과 표준 편차를 사용하여 스케일링합니다.

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(val_embeddings)
    test_embeddings_scaled = scaler.transform(pd.DataFrame(test_embeddings))
    y = val_target_labels
    df = pd.DataFrame(X_scaled)
    df["Target"] = val_target_list[:]
    class1_mean = df.groupby("Target").median().iloc[0,:]
    class2_mean = df.groupby("Target").median().iloc[1,:]
    class3_mean = df.groupby("Target").median().iloc[2,:]
    class4_mean = df.groupby("Target").median().iloc[3,:]

    class_means = np.array([class1_mean, class2_mean, class3_mean, class4_mean])

    # 테스트 데이터에 대해 분류 수행
    



    # 테스트 데이터에 대해 분류 수행
    predictions = classify_with_manhattan_distance(test_embeddings_scaled, class_means)
    

    # 정확도와 F1 스코어 계산
    accuracy = accuracy_score(test_target_labels, predictions)
    conf_matrix = confusion_matrix(test_target_labels, predictions)
    f1 = f1_score(test_target_labels, predictions, average='weighted')
    print(f'Final Test Accuracy: {accuracy:.2f}')
    print(f'Final Test F1 Score: {f1:.2f}')
    print(f'Conf_Matrix: {conf_matrix}')

    return f1