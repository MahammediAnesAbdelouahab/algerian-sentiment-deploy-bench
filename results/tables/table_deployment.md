| dataset | model | format | macro-F1 (%) | Δ vs native (pts) | size (MB) | peak RSS (MB) | cold start (s) | p50 (ms) | p95 (ms) | timing passes | p50 spread across passes (%) | p50 × fastest | speed-up vs PyTorch fp32 | throughput (texts/s) | CPU ms/text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| twifil_clean | SVM-char | scikit-learn | 60.8 | 0.0 | 3.0 | 204.0 | 1.95 | 1.05 | 1.45 | 4 | 62.8 | 1.0 |  | 4973.4 | 0.2 |
| twifil_clean | NB-char | scikit-learn | 60.4 | 0.0 | 4.6 | 196.0 | 2.6 | 1.54 | 1.86 | 4 | 58.4 | 1.5 |  | 4389.1 | 0.23 |
| twifil_clean | LR-word+char | scikit-learn | 61.0 | 0.0 | 3.4 | 206.0 | 2.11 | 2.71 | 3.93 | 4 | 7.6 | 2.6 |  | 2133.2 | 0.46 |
| twifil_clean | mE5-small | ONNX int8 | 69.8 | -0.01 | 129.1 | 473.0 | 3.24 | 14.1 | 32.33 | 4 | 58.4 | 13.5 | 3.0 | 20.6 | 48.01 |
| twifil_clean | mE5-small | ONNX int8 per-channel+RR | 69.9 | 0.1 | 129.3 | 474.0 | 2.58 | 15.24 | 33.17 | 4 | 29.6 | 14.5 | 2.8 | 24.3 | 40.89 |
| twifil_clean | mE5-small | ONNX int8 per-channel (no RR) | 70.3 | 0.49 | 129.3 | 519.0 | 3.32 | 21.26 | 51.84 | 1 | 13.5 | 20.3 | 2.0 | 19.5 | 50.15 |
| twifil_clean | mE5-small | ONNX fp32 | 69.8 | 0.0 | 465.3 | 875.0 | 4.33 | 25.31 | 58.64 | 1 | 11.5 | 24.2 | 1.7 | 16.0 | 61.7 |
| twifil_clean | DziriBERT | ONNX int8 per-channel+RR | 70.7 | -0.1 | 121.0 | 258.0 | 1.51 | 35.4 | 77.43 | 4 | 23.0 | 33.8 | 3.2 | 12.2 | 81.55 |
| twifil_clean | DziriBERT | ONNX int8 | 70.5 | -0.3 | 120.6 | 249.0 | 2.87 | 37.26 | 87.79 | 4 | 22.2 | 35.6 | 3.0 | 10.6 | 93.38 |
| twifil_clean | MARBERTv2 | ONNX int8 per-channel+RR | 69.4 | -1.0 | 159.0 | 304.0 | 1.43 | 38.05 | 103.21 | 4 | 7.6 | 36.3 | 3.1 | 9.9 | 100.7 |
| twifil_clean | MARBERTv2 | ONNX int8 | 67.8 | -2.54 | 158.6 | 292.0 | 4.3 | 40.13 | 107.3 | 4 | 23.1 | 38.3 | 2.9 | 8.5 | 116.72 |
| twifil_clean | mE5-small | PyTorch fp32 | 69.8 | 0.0 | 465.1 | 990.0 | 13.77 | 42.67 | 83.97 | 4 | 12.5 | 40.7 | 1.0 | 16.8 | 58.13 |
| twifil_clean | CAMeLBERT-DA | ONNX int8 per-channel+RR | 59.9 | -5.87 | 105.9 | 241.0 | 1.05 | 43.83 | 131.75 | 4 | 19.3 | 41.9 | 2.7 | 7.4 | 133.92 |
| twifil_clean | DziriBERT | ONNX int8 per-channel (no RR) | 69.3 | -1.52 | 121.0 | 328.0 | 1.95 | 45.15 | 106.9 | 1 | 1.5 | 43.1 | 2.5 | 9.2 | 105.73 |
| twifil_clean | CAMeLBERT-DA | ONNX int8 | 58.8 | -6.96 | 105.5 | 238.0 | 2.16 | 45.33 | 141.64 | 4 | 32.8 | 43.3 | 2.6 | 6.1 | 161.51 |
| twifil_clean | MARBERTv2 | ONNX int8 per-channel (no RR) | 16.5 | -53.9 | 159.0 | 387.0 | 3.89 | 48.15 | 119.3 | 1 | 6.0 | 46.0 | 2.4 | 7.9 | 124.39 |
| twifil_clean | CAMeLBERT-DA | ONNX int8 per-channel (no RR) | 22.9 | -42.95 | 105.9 | 402.0 | 2.62 | 55.64 | 170.71 | 1 | 29.3 | 53.1 | 2.1 | 5.9 | 165.38 |
| twifil_clean | DziriBERT | ONNX fp32 | 70.8 | 0.0 | 476.1 | 841.0 | 2.61 | 71.8 | 155.67 | 1 | 15.8 | 68.6 | 1.6 | 7.3 | 134.65 |
| twifil_clean | MARBERTv2 | ONNX fp32 | 70.4 | 0.0 | 624.0 | 991.0 | 5.27 | 73.34 | 175.82 | 1 | 2.5 | 70.0 | 1.6 | 5.9 | 168.43 |
| twifil_clean | CAMeLBERT-DA | ONNX fp32 | 65.8 | 0.0 | 417.1 | 862.0 | 4.36 | 109.18 | 252.12 | 1 | 3.7 | 104.3 | 1.1 | 4.4 | 222.96 |
| twifil_clean | DziriBERT | PyTorch fp32 | 70.8 | 0.0 | 475.9 | 891.0 | 11.73 | 112.24 | 206.68 | 4 | 13.9 | 107.2 | 1.0 | 7.7 | 128.07 |
| twifil_clean | CAMeLBERT-DA | PyTorch fp32 | 65.8 | 0.0 | 416.9 | 880.0 | 9.6 | 116.42 | 285.5 | 4 | 20.7 | 111.2 | 1.0 | 4.6 | 213.62 |
| twifil_clean | MARBERTv2 | PyTorch fp32 | 70.4 | 0.0 | 623.8 | 924.0 | 10.38 | 117.74 | 240.76 | 4 | 15.5 | 112.4 | 1.0 | 6.2 | 158.54 |
| youtube | SVM-char | scikit-learn | 76.8 | 0.0 | 10.0 | 261.0 | 3.4 | 1.02 | 1.62 | 4 | 31.0 | 1.0 |  | 2810.9 | 0.35 |
| youtube | NB-char | scikit-learn | 73.1 | 0.0 | 15.2 | 256.0 | 3.57 | 3.1 | 4.6 | 4 | 19.8 | 3.0 |  | 1578.2 | 0.6 |
| youtube | LR-word+char | scikit-learn | 75.9 | 0.0 | 14.0 | 304.0 | 3.48 | 6.37 | 8.72 | 4 | 55.9 | 6.2 |  | 1598.0 | 0.61 |
| youtube | mE5-small | ONNX int8 per-channel+RR | 75.8 | -0.34 | 129.3 | 474.0 | 1.94 | 15.54 | 50.52 | 4 | 18.6 | 15.2 | 2.9 | 14.0 | 70.65 |
| youtube | mE5-small | ONNX int8 per-channel (no RR) | 75.3 | -0.83 | 129.3 | 622.0 | 6.25 | 18.04 | 63.62 | 1 | 17.8 | 17.7 | 2.5 | 12.3 | 80.68 |
| youtube | mE5-small | ONNX int8 | 75.6 | -0.53 | 129.1 | 474.0 | 5.2 | 19.1 | 63.81 | 4 | 36.5 | 18.7 | 2.3 | 12.1 | 81.88 |
| youtube | mE5-small | ONNX fp32 | 76.1 | 0.0 | 465.3 | 956.0 | 11.59 | 25.33 | 80.94 | 1 | 46.5 | 24.8 | 1.8 | 9.0 | 110.65 |
| youtube | MARBERTv2 | ONNX int8 | 79.1 | 0.09 | 158.6 | 316.0 | 4.6 | 34.7 | 114.0 | 4 | 55.3 | 34.0 | 3.3 | 9.2 | 106.28 |
| youtube | DziriBERT | ONNX int8 per-channel+RR | 79.8 | -0.41 | 121.0 | 262.0 | 1.29 | 34.8 | 119.08 | 4 | 19.3 | 34.1 | 3.3 | 6.0 | 165.02 |
| youtube | MARBERTv2 | ONNX int8 per-channel+RR | 78.4 | -0.66 | 159.0 | 306.0 | 1.1 | 35.74 | 125.22 | 4 | 14.7 | 35.0 | 3.2 | 5.8 | 171.29 |
| youtube | DziriBERT | ONNX int8 | 79.7 | -0.44 | 120.6 | 268.0 | 2.89 | 36.66 | 131.19 | 4 | 26.9 | 35.9 | 3.1 | 4.8 | 205.45 |
| youtube | MARBERTv2 | ONNX int8 per-channel (no RR) | 17.5 | -61.56 | 159.0 | 527.0 | 8.26 | 39.79 | 141.97 | 1 | 42.0 | 39.0 | 2.9 | 4.9 | 198.85 |
| youtube | DziriBERT | ONNX int8 per-channel (no RR) | 79.0 | -1.18 | 121.0 | 476.0 | 1.86 | 41.42 | 145.78 | 1 | 2.5 | 40.6 | 2.7 | 4.7 | 207.37 |
| youtube | CAMeLBERT-DA | ONNX int8 per-channel+RR | 74.2 | -1.2 | 105.9 | 247.0 | 1.03 | 42.59 | 153.04 | 4 | 19.3 | 41.7 | 2.9 | 4.8 | 206.6 |
| youtube | mE5-small | PyTorch fp32 | 76.1 | 0.0 | 465.1 | 1001.0 | 17.33 | 44.41 | 111.29 | 4 | 21.1 | 43.5 | 1.0 | 10.5 | 93.31 |
| youtube | CAMeLBERT-DA | ONNX int8 | 72.2 | -3.21 | 105.5 | 268.0 | 3.42 | 44.65 | 162.49 | 4 | 21.4 | 43.7 | 2.7 | 4.0 | 244.95 |
| youtube | CAMeLBERT-DA | ONNX int8 per-channel (no RR) | 19.7 | -55.65 | 105.9 | 614.0 | 4.93 | 45.66 | 157.35 | 1 | 3.1 | 44.7 | 2.7 | 4.2 | 236.31 |
| youtube | MARBERTv2 | ONNX fp32 | 79.0 | 0.0 | 624.0 | 1115.0 | 14.87 | 67.71 | 199.42 | 1 | 3.0 | 66.3 | 1.7 | 3.6 | 271.36 |
| youtube | DziriBERT | ONNX fp32 | 80.2 | 0.0 | 476.1 | 956.0 | 6.93 | 80.05 | 215.67 | 1 | 23.7 | 78.4 | 1.4 | 3.3 | 298.57 |
| youtube | CAMeLBERT-DA | ONNX fp32 | 75.4 | 0.0 | 417.1 | 911.0 | 10.62 | 82.26 | 236.11 | 1 | 1.4 | 80.6 | 1.5 | 2.9 | 343.96 |
| youtube | DziriBERT | PyTorch fp32 | 80.2 | 0.0 | 475.9 | 903.0 | 13.25 | 113.33 | 266.05 | 4 | 32.2 | 111.0 | 1.0 | 3.5 | 276.63 |
| youtube | MARBERTv2 | PyTorch fp32 | 79.0 | 0.0 | 623.8 | 939.0 | 17.41 | 113.63 | 282.53 | 4 | 17.3 | 111.3 | 1.0 | 3.6 | 272.0 |
| youtube | CAMeLBERT-DA | PyTorch fp32 | 75.4 | 0.0 | 416.9 | 893.0 | 19.88 | 122.63 | 298.44 | 4 | 10.8 | 120.2 | 1.0 | 3.1 | 317.08 |
