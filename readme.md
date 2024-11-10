# Linear Transformer with Frequency Domain Filter for Multispectral and Hyperspectral Image Fusion

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## 📖 Abstract
The fusion of high-resolution multispectral images with low-resolution hyperspectral images, known as Multispectral and Hyperspectral Image Fusion (MHIF), is an effective method for hyperspectral image super-resolution. Recently, Transformer-based fusion methods have made significant strides in MHIF. However, the global self-attention mechanism in the Transformer increases computational complexity quadratically with the number of spectral tokens, and the model still exhibits shortcomings in capturing local spatial details within images. To tackle these challenges, this study presents a linear Transformer model with frequency domain filtering, referred to as LTFF. LTFF features a spatial-spectral linear attention mechanism that reduces self-attention complexity to $O(n)$. Additionally, it utilizes frequency domain filters to capture spatial-spectral dependencies, overcoming the limitations of spatial domain networks. Extensive experiments on three widely used benchmark datasets demonstrate that LTFF significantly lowers computational complexitywhile achieving state-of-the-art performance.

## 📦 Requirements
- Python 3.8+。
- PyTorch 1.6+
- CUDA 11.1+

## 📂Dataset
- [Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [Chikusei](https://naotoyokoya.com/Download.html)
- [Houston](https://hyperspectral.ee.uh.edu/?page_id=459)

## 🛠️Usage
Place the dataset in the dataset directory, and run the following command:
```bash
python main.py 
```

## 🔍 Contact

If you have any questions or suggestions, please submit an Issue or send a email to <lbs23@mails.jlu.edu.cn>.