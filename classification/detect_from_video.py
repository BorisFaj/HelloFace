"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from network.models import model_selection
from dataset.transform import xception_default_data_transforms

# SAM1 setup (Meta's official Segment Anything Model)
sam_checkpoint = 'sam_vit_b_01ec64.pth'  # Make sure this is downloaded
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.9,
    crop_n_layers=0,
    min_mask_region_area=200,
)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb

def preprocess_image(image, cuda=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
    preprocessed_image = preprocess_image(image, cuda)
    output = model(preprocessed_image)
    output = post_function(output)
    _, prediction = torch.max(output, 1)
    prediction = float(prediction.cpu().numpy())
    return int(prediction), output

def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True, conf_threshold=0.3, show=False, discriminator=None):
    print('Starting: {}'.format(video_path))
    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    face_detector = dlib.get_frontal_face_detector()

    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    if model_path is not None:
        model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model was given, initializing default model.')
    if cuda:
        model = model.cuda()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        ret, image = reader.read()
        if not ret:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        height, width = image.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps, (width, height))

        # output_frame = np.zeros_like(image)  # Black frame for background
        output_frame = image.copy()  # Copia la imagen original como fondo

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            for face in faces:
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y + size, x:x + size]

                face_segmented, mask = segment_face_with_sam(cropped_face)
                prediction, output = predict_with_model(face_segmented, model, cuda=cuda)

                # Penalizar confianza con proporción de píxeles visibles
                visible_ratio = np.sum(mask > 0) / mask.size
                conf = torch.max(output).item() * visible_ratio

                # Resize mask to original frame and paste segmented face
                mask_resized = cv2.resize(mask, (size, size))
                output_frame[y:y + size, x:x + size] = cv2.bitwise_or(
                    output_frame[y:y + size, x:x + size],
                    cv2.bitwise_and(cropped_face, cropped_face, mask=mask_resized)
                )

                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cropped_face = image[y:y + h, x:x + w]

                face_segmented, mask = segment_face_with_sam(cropped_face)
                prediction, output = predict_with_model(face_segmented, model, cuda=cuda)

                visible_ratio = np.sum(mask > 0) / mask.size
                conf = torch.max(output).item() * visible_ratio
            mask_resized
            # Aplicar directamente sin redimensionar
            mask_bbox_resized = mask  # ya tiene tamaño h x w
            face_resized = face_segmented  # ya tiene tamaño h x w

            # Fondo negro en el bbox antes de superponer la cara segmentada
            output_frame[y:y + h, x:x + w] = 0
            output_frame[y:y + h, x:x + w] = cv2.bitwise_or(
                output_frame[y:y + h, x:x + w],
                cv2.bitwise_and(face_resized, face_resized, mask=mask_bbox_resized)
            )

            label = 'real' if conf > conf_threshold else 'fake'
            color = (0, 255, 0) if conf > conf_threshold else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in output.detach().cpu().numpy()[0]]
            cv2.putText(output_frame, str(max(output_list)) + f' * ({conf:.2f}): {label} ', (x, y + h + 30),
                        font_face, font_scale, color, thickness, 2)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            torch.cuda.empty_cache()

        writer.write(output_frame)
        if show:
            cv2.imshow('test', output_frame)
            cv2.waitKey(1)
        if frame_num >= end_frame:
            break

    pbar.close()
    reader.release()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(join(output_path, video_fn)))
    else:
        print('Input video file was empty')
    cv2.destroyAllWindows()

def segment_face_with_sam(face_img):
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(rgb_face)

    if len(masks) == 0:
        return face_img, np.ones(face_img.shape[:2], dtype=np.uint8) * 255

    largest_mask = max(masks, key=lambda x: x['area'])['segmentation']
    mask_uint8 = largest_mask.astype(np.uint8) * 255
    face_segmented = cv2.bitwise_and(face_img, face_img, mask=mask_uint8)
    return face_segmented, mask_uint8

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--conf_threshold', type=int, default=0.3)
    p.add_argument('--show', default=False)
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))