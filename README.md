# 🎤 Speech Emotion Recognition 

用 LSTM、CNN、SVM、MLP 进行语音情感识别，Keras 实现。

改进了特征提取方式，识别准确率提高到了 80% 左右。




&nbsp;

## 📋 Environments

- Python 3.8
- Keras & TensorFlow 2


&nbsp;

## 📂 Project Structure

- `data/` - 数据集
- `models/` - 模型定义
- `notebooks/` - Jupyter Notebooks
- `scripts/` - 训练和评估脚本


&nbsp;

## Requirments

### Python

- [TensorFlow 2](https://github.com/tensorflow/tensorflow) / [Keras](https://github.com/keras-team/keras)：LSTM & CNN (`tensorflow.keras`)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)：SVM & MLP 模型，划分训练集和测试集
- [joblib](https://github.com/joblib/joblib)：保存和加载用 scikit-learn 训练的模型
- [librosa](https://github.com/librosa/librosa)：提取特征、波形图
- [SciPy](https://github.com/scipy/scipy)：频谱图
- [pandas](https://github.com/pandas-dev/pandas)：加载特��
- [Matplotlib](https://github.com/matplotlib/matplotlib)：绘图
- [NumPy](https://github.com/numpy/numpy)

### Tools

- [可选] [Opensmile](https://github.com/naxingyu/opensmile)：提取特征


&nbsp;

## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   英文，24 个人（12 名男性，12 名女性）的大约 1500 个音频，表达了 8 种不同的情绪（第三位数字表示情绪类别）：01 = neutral，02 = calm，03 = happy，04 = sad，05 = angry，06 = fearful，07 = disgust，08 = surprised。

2. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

   英文，4 个人（男性）的大约 500 个音频，表达了 7 种不同的情绪（第一个字母表示情绪类别）：a = anger，d = disgust，f = fear，h = happiness，n = neutral，sa = sadness，su = surprise。

3. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   德语，10 个人（5 名男性，5 名女性）的大约 500 个音频，表达了 7 种不同的情绪（倒数第二个字母表示情绪类别）：N = neutral，W = angry，A = fear，F = happy，T = sad，E = disgust，L = boredom。

4. CASIA

   汉语，4 个人（2 名男性，2 名女性）的大约 1200 个音频，表达了 6 种不同的情绪：neutral，happy，sad，angry，fearful，surprised。


&nbsp;

## 🚀 Getting Started

1. 克隆仓库：
   ```sh
   git clone https://github.com/qianxingyucode/SER.git
   cd SER
   ```

&nbsp;

## Usage

### Prepare

安装依赖：

```python
pip install -r requirements.txt
```

（可选）安装 [Opensmile](https://github.com/naxingyu/opensmile)。

&nbsp;

### Configuration

在 [`configs/`](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/master/configs) 文件夹中的配置文件（YAML）里配置参数。

其中 Opensmile 标准特征集目前只支持：

- `IS09_emotion`：[The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf)，384 个特征；
- `IS10_paraling`：[The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf)，1582 个特征；
- `IS11_speaker_state`：[The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf)，4368 个特征；
- `IS12_speaker_trait`：[The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf)，6125 个特征；
- `IS13_ComParE`：[The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf)，6373 个特征；
- `ComParE_2016`：[The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf)，6373 个特征。

如果需要用其他特征集，可以自行修改 [`extract_feats/opensmile.py`](extract_feats/opensmile.py) 中的 `FEATURE_NUM` 项。

&nbsp;

### Preprocess

首先需要提取数据集中音频的特征并保存到本地。Opensmile 提取的特征会被保存在 `.csv` 文件中，librosa 提取的特征会被保存在 `.p` 文件中。

```python
python preprocess.py --config configs/example.yaml
```
其中，`configs/example.yaml` 是你的配置文件路径。

&nbsp;

### Train

数据集路径可以在 [`configs/`](configs) 中配置，相同情感的音频放在同一个文件夹里（可以参考 [`utils/files.py`](utils/files.py) 整理数据），如：

```
└── datasets
    ├── angry
    ├── happy
    ├── sad
    ...
```

然后：

```python
python train.py --config configs/example.yaml
```

&nbsp;

### Predict

用训练好的模型来预测指定音频的情感。[`checkpoints/`](checkpoints)里有一些已经训练好的模型。

```python
python predict.py --config configs/example.yaml
```


&nbsp;

### Functions

#### Radar Chart

画出预测概率的雷达图。

来源：[Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
import utils

"""
Args:
    data_prob (np.ndarray): 概率数组
    class_labels (list): 情感标签
"""
utils.radar(data_prob, class_labels)
```

&nbsp;

#### Play Audio

播放一段音频

```python
import utils

utils.play_audio(file_path)
```

&nbsp;

#### Plot Curve

画训练过程的准确率曲线和损失曲线。

```python
import utils

"""
Args:
    train (list): 训练集损失值或准确率数组
    val (list): 测试集损失值或准确率数组
    title (str): 图像标题
    y_label (str): y 轴标题
"""
utils.curve(train, val, title, y_label)
```

&nbsp;

#### Waveform

画出音频的波形图。

```python
import utils

utils.waveform(file_path)
```

&nbsp;

#### Spectrogram

画出音频的频谱图。

```python
import utils

utils.spectrogram(file_path)
```


&nbsp;

## Other Contributors

- [@Zhaofan-Su](https://github.com/Zhaofan-Su)
- [@Guo Hui](https://github.com/guohui15661353950)
