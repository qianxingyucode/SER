import os
import numpy as np
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils

def predict(config, audio_path: str, model) -> None:
    """
    预测音频情感

    Args:
        config: 配置项
        audio_path (str): 要预测的音频路径
        model: 加载的模型
    """

    utils.play_audio(audio_path)
    # utils.waveform(audio_path)
    # utils.spectrogram(audio_path)
    if config.feature_method == 'o':
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(config, audio_path, train=False)
        test_feature = of.load_feature(config, train=False)
    elif config.feature_method == 'l':
        test_feature = lf.get_data(config, audio_path, train=False)
    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print('Recogntion: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    utils.radar(result_prob, config.class_labels)


if __name__ == '__main__':
    audio_path = 'D:/桌面/SER/datasets/CASIA/6/happy/201-happy-liuchanhg.wav'
    # D:\桌面\SER\datasets\CASIA\6\angry\201-angry-liuchanhg.wav
    # audio_path = 'datasets/CASIA/6/happy/201-happy-wangzhe.wav'
    # audio_path = 'datasets/CASIA/6/sad/201-sad-wangzhe.wav'
    config = utils.parse_opt()
    model = models.load(config)
    predict(config, audio_path, model)
