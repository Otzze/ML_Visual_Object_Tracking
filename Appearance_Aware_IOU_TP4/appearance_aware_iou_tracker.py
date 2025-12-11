
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import cv2
from KalmanFilter import KalmanFilter
import configparser
import onnxruntime

class FeatureExtractor:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.roi_width = 64
        self.roi_height = 128
        self.roi_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.roi_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_patch(self, im_crops):
        roi_input = cv2.resize(im_crops, (self.roi_width, self.roi_height))
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        roi_input = (roi_input.astype(np.float32) / 255. - self.roi_means) / self.roi_stds
        roi_input = np.transpose(roi_input, (2, 0, 1))
        return np.expand_dims(roi_input, axis=0)

    def __call__(self, patch):
        processed_patch = self.preprocess_patch(patch)
        features = self.session.run([self.output_name], {self.input_name: processed_patch})[0]
        return features.flatten() / np.linalg.norm(features.flatten())


def load_detections(det_file_path):
    detections = {}
    with open(det_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame = int(parts[0])
            if frame not in detections:
                detections[frame] = []
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            conf = float(parts[6])
            detections[frame].append(np.array([bb_left, bb_top, bb_width, bb_height, conf]))
    return detections

def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_dt_from_seqinfo(seq_info_path):
    config = configparser.ConfigParser()
    with open(seq_info_path, 'r') as f:
        config.read_string(f.read())
    return 1.0 / int(config['Sequence']['frameRate'])

def main():
    seq_name = "ADL-Rundle-6"
    img_dir = os.path.join(seq_name, "img1")
    det_file = os.path.join(seq_name, "det", "Yolov5s", "det.txt")
    seq_info_file = os.path.join(seq_name, "seqinfo.ini")
    model_path = "reid_osnet_x025_market1501.onnx"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    dt = get_dt_from_seqinfo(seq_info_file)
    iou_threshold = 0.3
    max_age = 5
    alpha = 0.6 

    feature_extractor = FeatureExtractor(model_path)
    detections = load_detections(det_file)
    
    tracks = []
    next_track_id = 1
    results = []

    num_frames = len(os.listdir(img_dir))

    for frame_num in range(1, num_frames + 1):
        print(f"Processing frame {frame_num}/{num_frames}")
        
        img_path = os.path.join(img_dir, f"{frame_num:06d}.jpg")
        frame_img = cv2.imread(img_path)

        for track in tracks:
            track['kf'].predict()
            predicted_state = track['kf'].Xk
            predicted_cx = predicted_state[0, 0]
            predicted_cy = predicted_state[1, 0]
            w, h = track['w'], track['h']
            track['bbox'] = [predicted_cx - w/2, predicted_cy - h/2, w, h]

        current_detections = detections.get(frame_num, [])
        detection_features = []
        for det in current_detections:
            x, y, w, h = map(int, det[:4])
            patch = frame_img[y:y+h, x:x+w]
            if patch.size > 0:
                detection_features.append(feature_extractor(patch))
            else:
                detection_features.append(np.zeros(512))


        if len(tracks) > 0 and len(current_detections) > 0:
            cost_matrix = np.zeros((len(tracks), len(current_detections)))
            for i, track in enumerate(tracks):
                for j, det in enumerate(current_detections):
                    iou_cost = 1 - iou(track['bbox'], det[:4])
                    app_cost = cosine(track['features'], detection_features[j])
                    cost_matrix[i, j] = alpha * iou_cost + (1 - alpha) * app_cost
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_indices = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_detections = list(range(len(current_detections)))

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1 - iou_threshold:
                    det = current_detections[c]
                    w, h = det[2], det[3]
                    cx, cy = det[0] + w/2, det[1] + h/2
                    z = np.array([[cx], [cy]])

                    tracks[r]['kf'].update(z)
                    updated_state = tracks[r]['kf'].Xk
                    updated_cx, updated_cy = updated_state[0, 0], updated_state[1, 0]
                    
                    tracks[r]['bbox'] = [updated_cx - w/2, updated_cy - h/2, w, h]
                    tracks[r]['w'], tracks[r]['h'] = w, h
                    tracks[r]['age'] = 0
                    tracks[r]['features'] = 0.1 * detection_features[c] + 0.9 * tracks[r]['features']
                    
                    matched_indices.append(c)
                    if r in unmatched_tracks:
                        unmatched_tracks.remove(r)
            
            unmatched_detections = [d for d in unmatched_detections if d not in matched_indices]

        else:
            unmatched_tracks = list(range(len(tracks)))
            unmatched_detections = list(range(len(current_detections)))

        for track_idx in unmatched_tracks:
            tracks[track_idx]['age'] += 1
        
        tracks = [t for t in tracks if t['age'] <= max_age]

        for det_idx in unmatched_detections:
            det = current_detections[det_idx]
            w, h = det[2], det[3]
            cx, cy = det[0] + w/2, det[1] + h/2

            kf = KalmanFilter(dt=dt, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
            kf.Xk = np.array([[cx], [cy], [0], [0]])

            new_track = {
                'id': next_track_id, 'kf': kf, 'bbox': det[:4], 'age': 0, 'w': w, 'h': h,
                'features': detection_features[det_idx]
            }
            tracks.append(new_track)
            next_track_id += 1

        for track in tracks:
            x, y, w, h = track['bbox']
            results.append([frame_num, track['id'], x, y, w, h, 1, -1, -1, -1])

    output_file = os.path.join(output_dir, f"{seq_name}.txt")
    with open(output_file, 'w') as f:
        for res in results:
            f.write(','.join(map(str, res)) + '\n')
    print(f"Results saved to {output_file}")

    print("Generating video...")
    video_writer = None
    for frame_num in range(1, num_frames + 1):
        img_path = os.path.join(img_dir, f"{frame_num:06d}.jpg")
        frame = cv2.imread(img_path)
        if frame is None: continue

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(os.path.join(output_dir, f"{seq_name}.mp4"), fourcc, 1.0/dt, (frame.shape[1], frame.shape[0]))
        
        frame_results = [r for r in results if r[0] == frame_num]
        for res in frame_results:
            track_id, x, y, w, h = res[1], *res[2:6]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        video_writer.write(frame)

    if video_writer is not None: video_writer.release()
    print(f"Video saved to {os.path.join(output_dir, f'{seq_name}.mp4')}")

if __name__ == "__main__":
    main()
