### Introduction

Encoder-only transformer for embroidery gesture classification. Google’s Mediapipe Handlandmarks were used to extract the most prominent, frame-wise, hand landmarks from two camera perspectives of an embroidery process (top-down and bottom-up). These landmarks were converted into temporal matrices, of the form (T, 126) where T represents all recordings within a current frame and 126 is formed by the concatenation of flattened Handlandmark matrices with an original shape of (21,3). The resulting dataset was used to train an encoder-only transformer to act as a binary classifier, to inform a real-time software-based running accuracy score based on gesture performance. Inference has been configured to simulate dual camera streamed video footage in a manner that should be easily switched for live dual camera input. The model was trained and inference has been conducted using cuda 11.8 and python 3.11.11.


### Installation

1. Clone this repo:
```bash
git clone https://github.com/x-ix/Embroidery_Gesture_Recognition.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```

### Usage
```
Usage:
        python inference.py

```


### Important Notes

- During inference the model recieves and analyses matrices of the form (1,126), this has been implemented in inference.py already.

- norm_stats.pt contains the mean and std used during training and has been applied to input matrices just prior to inference, if adapting the algorithm please retain this normalisation for accurate model performance.

- I haven't tested this using cuda nor python versions outside of 11.8 and 3.11.11 respectively, thus cannot comment on the programs functionality outside of those environments.



### Miscellaneous
Contents of [requirements.txt](requirements.txt):
```
--index-url https://download.pytorch.org/whl/cu118
torch==2.0.1 
torchvision==0.15.2
torchaudio==2.0.2

--extra-index-url https://pypi.org/simple
mediapipe==0.10.21
numpy==1.26.4
opencv-python==4.11.0.86
```


### Closing Notes
Performance is not stellar but is being worked on.
