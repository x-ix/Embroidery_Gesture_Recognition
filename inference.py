import torch
import torch.nn as nn
import math
import warnings
import torch.nn.functional as F
import time

import mediapipe as mp
import numpy as np
import cv2

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#mediapipe config
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#mediapipe Constants
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
MS_IN_SECOND = 1000
HANDS = 1

# Comment out if streaming from webcam
video_path_1 = "Top-View.mp4"
video_path_2 = "Bottom-View.mp4"


#POSITIONAL ENCODING PARAMETERS
DROPOUT = 0.0
MAX_LEN = 5000

#TRANSFORMER PARAMETERS
INPUT_DIM = 126
D_MODEL = 64
NUM_HEADS = 8
NUM_LAYERS = 4 #was 2
NUM_CLASSES = 2
FF_DIM_MULT = D_MODEL * 4

#mean and std, import to apply to input dim before inference
stats = torch.load('norm_stats.pt')
mean, std = stats['mean'], stats['std']



# --- Transformer model and PE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()  # shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # Even indices use sin, odd use cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=FF_DIM_MULT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x, pad_mask):
        """
        x: (batch_size, seq_len, input_dim)
        pad_mask: (batch_size, seq_len) with True at padding positions
        """
        # 1. Input projection and positional encoding
        x = self.input_proj(x) * math.sqrt(self.d_model)  # scale as in original transformer
        x = self.pos_encoder(x)

        # 2. Apply TransformerEncoder with padding mask
        #    The mask (src_key_padding_mask) has shape (batch, seq_len) per docs:contentReference[oaicite:4]{index=4}.
        out = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        
        # 3. Mean pooling over the time dimension (ignore padded positions)
        mask = (~pad_mask).unsqueeze(2)  # shape (batch, seq_len, 1), True for valid tokens
        out = out * mask  # zero out padded positions
        lengths = mask.sum(dim=1)  # sum of ones = length of each sequence
        # Avoid division by zero
        lengths[lengths == 0] = 1
        out = out.sum(dim=1) / lengths  # (batch, d_model)
        # 4. Classifier
        logits = self.fc_out(out)  # (batch, num_classes)
        return logits



# --- Function for inference ---
class StreamPredictor:
    def __init__(self, model, device, window_size=None):

        self.model = model.to(device)
        self.device = device
        self.window_size = window_size
        self.sequence = torch.empty((0, 126)).to(device)  # Start empty

    def reset(self):
        """Clear the accumulated sequence."""
        self.sequence = torch.empty((0, 126)).to(self.device)

    def update(self, new_vector):
        """
        Takes in matrix of form (1, 126),
        Returns pred_class (int), prob_class1 (float)
        """
        assert new_vector.shape == (1, 126), "Input must be of shape (1, 126)"
        self.sequence = torch.cat([self.sequence, new_vector.to(self.device)], dim=0)

        # Apply sliding window if needed
        if self.window_size is not None and self.sequence.size(0) > self.window_size:
            self.sequence = self.sequence[-self.window_size:]

        # Prepare model input
        seq_tensor = self.sequence.unsqueeze(0)  # (1, L, 126)
        pad_mask = torch.zeros((1, seq_tensor.size(1)), dtype=torch.bool, device=self.device)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(seq_tensor, pad_mask)  # (1, 2)
            probs = F.softmax(logits, dim=1)
            prob1 = probs[0, 1].item()
            pred_class = torch.argmax(probs, dim=1).item()
        
        return pred_class, prob1



#cv2 capture initialisation and fps retrieval 
def initialise_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps



# --- main ---
def main():
    # --- Loading model for inference ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier().to(device)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    predictor = StreamPredictor(model, device, window_size=100)



    # --- Mediapipe model --
    model = model_path
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=HANDS,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1
    )



    #Setting up two capture devices to simulate two input streams
    cap1, fps1 = initialise_video_capture(video_path_1)
    cap2, fps2 = initialise_video_capture(video_path_2)

    with HandLandmarker.create_from_options(options) as landmarker1, \
     HandLandmarker.create_from_options(options) as landmarker2:
        
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break

            frame_number1 = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
            frame_number2 = int(cap2.get(cv2.CAP_PROP_POS_FRAMES))

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1)
            mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame2)

            timestamp_ms1 = ((frame_number1 / fps1) * MS_IN_SECOND) - (MS_IN_SECOND / fps1)
            timestamp_ms2 = ((frame_number2 / fps2) * MS_IN_SECOND) - (MS_IN_SECOND / fps2)

            start_time = time.time()
            detection_result1 = landmarker1.detect_for_video(mp_image1, int(timestamp_ms1))
            detection_result2 = landmarker2.detect_for_video(mp_image2, int(timestamp_ms2))

            if detection_result1.hand_landmarks and detection_result2.hand_landmarks:
                hand1 = detection_result1.hand_landmarks[0]  # assuming one hand
                hand2 = detection_result2.hand_landmarks[0]  # assuming one hand

                #Handlandmarks come in the form (1, 3) for 21 landmarks resulting in a matrix of (21, 3) which is then flattened to (1,63)
                matrix_entry1 = np.array([[lm.x, lm.y, lm.z] for lm in hand1], dtype=np.float32).flatten()
                matrix_entry2 = np.array([[lm.x, lm.y, lm.z] for lm in hand2], dtype=np.float32).flatten()

                # Convert to tensors without copying
                tensor1 = torch.from_numpy(matrix_entry1).unsqueeze(0)  # shape (1, 63)
                tensor2 = torch.from_numpy(matrix_entry2).unsqueeze(0)  # shape (1, 63)

                # Concatenate (Top-View on left concatenated with Bottom-View on right) to from a matrix of shape (1, 63 + 63) = (1, 126)
                combined = torch.cat([tensor1, tensor2], dim=1)  # shape (1, 126)
                normalized = (combined - mean) / std # normalise

                
                pred_class, prob1 = predictor.update(normalized)
                end_time = time.time()

                inference_time_ms = (end_time - start_time) * 1000  
                
                print(f"Prediction: {pred_class}, P(class=1): {prob1:.4f}, Inference time: {inference_time_ms:.2f} ms")

    cap1.release()
    cap2.release()
 

if __name__ == "__main__":
    main()








