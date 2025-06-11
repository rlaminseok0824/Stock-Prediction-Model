import os
from pathlib import Path
from typing import Union, Tuple
import warnings

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from yacs.config import CfgNode as CN

from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class ForecastingDataset(Dataset):
    def __init__(
        self, 
        data_dir: Union[str, Path], 
        n_var: int,
        seq_len: int, 
        label_len: int, 
        pred_len: int, 
        features: str,
        timeenc: int,
        freq: str,
        date_idx: int,
        target_start_idx: int,
        scale="standard", 
        split="train", 
        train_ratio=0.7,
        test_ratio=0.2
        ):
        assert split in ('train', 'val', 'test')
        
        self.data_dir = data_dir
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.date_idx = date_idx
        self.target_start_idx = target_start_idx
        
        self.scale = scale
        self.split = split
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.train, self.val, self.test, self.train_stamp,  self.val_stamp, self.test_stamp = self._load_data()
        assert self.train.shape[1] == n_var
        
        self._normalize_data()
        # self.print_data_stats()

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        raise NotImplementedError

    def _split_data(self, df_raw: pd.DataFrame) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        assert 0.0 < self.train_ratio < 1.0 and 0.0 < self.test_ratio < 1.0 and self.train_ratio + self.test_ratio <= 1.0
        
        data = df_raw[df_raw.columns[1:]].values
        train_len = int(len(data) * self.train_ratio)
        test_len = int(len(data) * self.test_ratio)
        val_len = len(data) - train_len - test_len
        
        train_start = 0
        train_end = train_len
    
        val_start = train_len - self.seq_len
        val_end = train_len + val_len
        
        test_start = train_len + val_len - self.seq_len
        test_end = len(data)
        
        train = data[train_start:train_end]
        val = data[val_start:val_end]
        test = data[test_start:test_end]
        
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            # df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            # df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            # df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['year'] = df_stamp.date.dt.year
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        train_stamp = data_stamp[train_start:train_end]
        val_stamp = data_stamp[val_start:val_end]
        test_stamp = data_stamp[test_start:test_end]
        
        return train, val, test, train_stamp, val_stamp, test_stamp

    def _normalize_data(self):
        if self.scale == "standard":
            scaler = StandardScaler()
        elif self.scale == "min-max":
            scaler = MinMaxScaler()
        elif self.scale == "min-max_fixed":
            return
        else:
            raise ValueError

        self.train = scaler.fit_transform(self.train)
        self.val = scaler.transform(self.val)
        self.test = scaler.transform(self.test)

    def print_data_stats(self):
        if self.split == 'train':
            print(f"Train data shape: {self.train.shape}, mean: {np.mean(self.train, axis=0)}, std: {np.std(self.train, axis=0)}")
        elif self.split == 'val':
            print(f"Validation data shape: {self.val.shape}, mean: {np.mean(self.val, axis=0)}, std: {np.std(self.val, axis=0)}")
        elif self.split == 'test':
            print(f"Test data shape: {self.test.shape}, mean: {np.mean(self.test, axis=0)}, std: {np.std(self.test, axis=0)}")

    def __len__(self):
        if self.split == "train":
            return len(self.train) - self.seq_len - self.pred_len + 1
        elif self.split == "val":
            return len(self.val) - self.seq_len - self.pred_len + 1
        elif self.split == "test":
            return len(self.test) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        if self.split == "train":
            data, stamp = self.train, self.train_stamp
        elif self.split == 'val':
            data, stamp = self.val, self.val_stamp
        elif self.split == 'test':
            data, stamp = self.test, self.test_stamp
        
        enc_start_idx = index
        enc_end_idx = index + self.seq_len
        dec_start_idx = enc_end_idx - self.label_len
        dec_end_idx = dec_start_idx + self.label_len + self.pred_len
        
        enc_window = data[enc_start_idx:enc_end_idx]
        enc_window_stamp = stamp[enc_start_idx:enc_end_idx]
        
        dec_window = data[dec_start_idx:dec_end_idx]
        dec_window_stamp = stamp[dec_start_idx:dec_end_idx]
        
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp


class Weather(ForecastingDataset):
    def __init__(
        self, 
        data_dir: Union[str, Path], 
        n_var: int,
        seq_len: int, 
        label_len: int, 
        pred_len: int, 
        features: str,
        timeenc: int,
        freq: str,
        date_idx: int,
        target_start_idx: int,
        scale="standard", 
        split="train", 
        train_ratio=0.7,
        test_ratio=0.2
        ):
        super(Weather, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'weather.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp
    
    
class Illness(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(Illness, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'illness.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp
    
    
class Electricity(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(Electricity, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'electricity.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp


class Traffic(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(Traffic, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'traffic.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp


class Exchange(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(Exchange, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'exchange_rate.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp


class ETTh1(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(ETTh1, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'ETTh1.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp


class ETTh2(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(ETTh2, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'ETTh2.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp


class ETTm1(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(ETTm1, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'ETTm1.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp


class ETTm2(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(ETTm2, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'ETTm2.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp
    

class Stock(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2
    ):
        super(Stock, self).__init__(data_dir, n_var, seq_len, label_len, pred_len, features, timeenc, freq, date_idx, target_start_idx, scale, split, train_ratio, test_ratio)

    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'stock.csv'))

        assert df_raw.columns[self.date_idx] == 'date'
        
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_raw)
        
        return train, val, test, train_stamp, val_stamp, test_stamp
    
class StockPctChange(ForecastingDataset):
    def __init__(
    self, 
    data_dir: Union[str, Path], 
    n_var: int,
    seq_len: int, 
    label_len: int, 
    pred_len: int, 
    features: str,
    timeenc: int,
    freq: str,
    date_idx: int,
    target_start_idx: int,
    scale="standard", 
    split="train", 
    train_ratio=0.7,
    test_ratio=0.2,
    target: str = 'close',
    percent: str = 100
    ):  
        # 추가 속성들
        self.target = target
        self.percent = percent
        
        # 부모 클래스 초기화
        super().__init__(
            data_dir=data_dir,
            n_var=n_var,  # 임시값, _load_data에서 실제 확인
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            features=features,
            timeenc=timeenc,
            freq=freq,
            date_idx=date_idx,  # 첫 번째 컬럼이 date
            target_start_idx=target_start_idx,  # date 다음부터 시작
            scale=scale,
            split=split,
            train_ratio=train_ratio,
            test_ratio=test_ratio
        )
        
        # 디버그 정보 출력
        current_data = self.get_current_data()
        print(f"[DEBUG] {self.split} data shape: {current_data.shape}")
        print(f"[DEBUG] seq_len: {self.seq_len}, pred_len: {self.pred_len}")
        print(f"[DEBUG] __len__ result: {len(self)}")
        print(f"[DEBUG] Expected steps for test: {len(self) - self.pred_len - 1 if self.split == 'test' else 'N/A'}")
    
    def _load_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        """데이터 로드 및 변화율 계산"""
        df_raw = pd.read_csv(os.path.join(self.data_dir, 'stock.csv'))

        
        # 컬럼 재배열: date, features, target 순서
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # 날짜 컬럼이 첫 번째, 타겟 컬럼이 마지막인지 확인
        assert df_raw.columns[self.date_idx] == 'date'
        assert df_raw.columns[-1] == self.target

        # 특성 선택
        if self.features == 'M' or self.features == 'MS':
            # 모든 특성 사용 (날짜 제외)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 타겟만 사용
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw[df_raw.columns[1:]]
        
        # 변화율 계산
        eps = 1e-6
        data_values = df_data.values
        if data_values.shape[0] < 2:
            pct_data = np.zeros((1, data_values.shape[1]))
        else:
            pct_data = (data_values[1:] - data_values[:-1]) / (data_values[:-1] + eps)
        
        # 날짜 데이터도 맞춰서 조정 (변화율 계산으로 인해 1개 줄어듦)
        df_raw_adjusted = df_raw.iloc[1:].reset_index(drop=True)
        
        # 부모 클래스의 _split_data 사용을 위해 DataFrame 재구성
        df_for_split = pd.concat([
            df_raw_adjusted[['date']].reset_index(drop=True),
            pd.DataFrame(pct_data, columns=df_data.columns)
        ], axis=1)
        
        # 부모 클래스의 분할 메서드 사용
        train, val, test, train_stamp, val_stamp, test_stamp = self._split_data(df_for_split)
        
        # 원본 종가 데이터도 저장 (복원용)
        original_close_data = df_raw[[self.target]].values[1:]  # 변화율과 맞춰서 1개 줄임
        
        # 각 split에 맞게 원본 종가 데이터 분할
        train_len = int(len(pct_data) * self.train_ratio)
        test_len = int(len(pct_data) * self.test_ratio)
        
        train_start, train_end = 0, train_len
        val_start = train_len - self.seq_len
        val_end = train_len + (len(pct_data) - train_len - test_len)
        test_start = train_len + (len(pct_data) - train_len - test_len) - self.seq_len
        test_end = len(pct_data)
        
        self.original_close_train = original_close_data[train_start:train_end]
        self.original_close_val = original_close_data[val_start:val_end]
        self.original_close_test = original_close_data[test_start:test_end]
        
        return train, val, test, train_stamp, val_stamp, test_stamp
    
    def get_current_data(self):
        """현재 split에 해당하는 데이터 반환"""
        if self.split == "train":
            return self.train
        elif self.split == "val":
            return self.val
        else:
            return self.test
    
    def get_current_original_close(self):
        """현재 split에 해당하는 원본 종가 데이터 반환"""
        if self.split == "train":
            return self.original_close_train
        elif self.split == "val":
            return self.original_close_val
        else:
            return self.original_close_test
    
    def __getitem__(self, index):
        """데이터 아이템 반환"""
        current_data = self.get_current_data()
        current_stamp = getattr(self, f"{self.split}_stamp")
        current_original_close = self.get_current_original_close()
        
        # 단일 시계열로 처리 (호환성을 위해)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 시퀀스 데이터 추출
        seq_x = current_data[s_begin:s_end]  # 모든 특성 포함
        seq_y = current_data[r_begin:r_end]  # 모든 특성 포함
        seq_x_mark = current_stamp[s_begin:s_end]
        seq_y_mark = current_stamp[r_begin:r_end]
        
        # 원본 종가 데이터 (복원용) - target 컬럼만
        target_idx = -1 if self.features in ['M', 'MS'] else 0
        close_y_original = current_original_close[r_begin:r_end]
        
        return seq_x, seq_x_mark, seq_y, seq_y_mark
    
    @staticmethod
    def restore_predicted_close(predicted, actual, actual_close):
        """예측된 변화율을 종가로 복원"""
        prev_close = actual_close / (1 + actual)
        predicted_close = prev_close * (1 + predicted)
        return predicted_close
        
        
def build_dataset(cfg, split):
    data_name = cfg.DATA.NAME
    dataset_config = dict(
        data_dir=os.path.join(cfg.DATA.BASE_DIR, data_name),
        n_var=cfg.DATA.N_VAR,
        seq_len=cfg.DATA.SEQ_LEN,
        label_len=cfg.DATA.LABEL_LEN,
        pred_len=cfg.DATA.PRED_LEN,
        features=cfg.DATA.FEATURES,
        timeenc=cfg.DATA.TIMEENC,
        freq=cfg.DATA.FREQ,
        date_idx=cfg.DATA.DATE_IDX,
        target_start_idx=cfg.DATA.TARGET_START_IDX,
        scale=cfg.DATA.SCALE,
        split=split,
        train_ratio=cfg.DATA.TRAIN_RATIO,
        test_ratio=cfg.DATA.TEST_RATIO,
    )
        
    if data_name == "weather":
        dataset = Weather(**dataset_config)
    elif data_name == 'illness':
        dataset = Illness(**dataset_config)
    elif data_name == 'electricity':
        dataset = Electricity(**dataset_config)
    elif data_name == 'traffic':
        dataset = Traffic(**dataset_config)
    elif data_name == 'exchange_rate':
        dataset = Exchange(**dataset_config)
    elif data_name == 'ETTh1':
        dataset = ETTh1(**dataset_config)
    elif data_name == 'ETTh2':
        dataset = ETTh2(**dataset_config)
    elif data_name == 'ETTm1':
        dataset = ETTm1(**dataset_config)
    elif data_name == 'ETTm2':
        dataset = ETTm2(**dataset_config)
    elif data_name == 'stock':
        dataset = Stock(**dataset_config)
    elif data_name == 'stock_change':
        dataset = StockPctChange(**dataset_config, target='close', percent=100)
    else:
        raise ValueError

    return dataset


def update_cfg_from_dataset(cfg: CN, dataset_name: str):
    cfg.DATA.NAME = dataset_name
    if dataset_name == 'weather':
        n_var = 21
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 12  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'illness':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 6  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'electricity':
        n_var = 321
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'traffic':
        n_var = 862
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'exchange_rate':
        n_var = 8
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 6  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTh1':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTh2':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 24  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTm1':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 12  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'ETTm2':
        n_var = 7
        cfg.DATA.N_VAR = n_var
        cfg.DATA.FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA.PERIOD_LEN = 12  #! for SAN
        cfg.DATA.TRAIN_RATIO = 0.6
        cfg.DATA.TEST_RATIO = 0.2
        
        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'stock_change':
        n_var = 5
        cfg.DATA.N_VAR = n_var
        cfg.DATA_FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA_PERIOD_LEN = 6 #! for SAN
        cfg .DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2

        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    elif dataset_name == 'stock':
        n_var = 5
        cfg.DATA.N_VAR = n_var
        cfg.DATA_FEATURES = 'M'
        cfg.DATA.TARGET_START_IDX = 0
        cfg.DATA_PERIOD_LEN = 6 #! for SAN
        cfg .DATA.TRAIN_RATIO = 0.7
        cfg.DATA.TEST_RATIO = 0.2

        cfg.MODEL.enc_in = n_var
        cfg.MODEL.dec_in = n_var
        cfg.MODEL.c_out = n_var
    else:
        raise ValueError
