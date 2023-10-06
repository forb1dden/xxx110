from typing import Any, List, Callable
import cv2
import threading
import numpy
import onnxruntime

import facefusion.globals
from facefusion import wording
from facefusion.core import update_status
from facefusion.face_analyser import get_many_faces
from facefusion.typing import Frame, Face, ProcessMode
from facefusion.utilities import conditional_download, resolve_relative_path, is_image, is_video

FRAME_PROCESSOR = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'FACEFUSION.FRAME_PROCESSOR.FACE_ENHANCER_PRO'


def get_frame_processor() -> Any:
	global FRAME_PROCESSOR

	with THREAD_LOCK:
		if FRAME_PROCESSOR is None:
			model_path = resolve_relative_path('../.assets/models/GFPGANv1.4.onnx')
			FRAME_PROCESSOR = onnxruntime.InferenceSession(model_path, providers = facefusion.globals.execution_providers)
	return FRAME_PROCESSOR


def clear_frame_processor() -> None:
	global FRAME_PROCESSOR

	FRAME_PROCESSOR = None


def pre_check() -> bool:
	download_directory_path = resolve_relative_path('../.assets/models')
	conditional_download(download_directory_path, [ 'https://github.com/facefusion/facefusion-assets/releases/download/models/GFPGANv1.4.onnx' ])
	return True


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(facefusion.globals.target_path) and not is_video(facefusion.globals.target_path):
		update_status(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
		return False
	if mode == 'output' and not facefusion.globals.output_path:
		update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
		return False
	return True


def post_process() -> None:
	clear_frame_processor()


def norm_crop2(img, landmark, image_size, enable_padding=True):
	lm = numpy.array(landmark)
	eye_left = lm[0]
	eye_right = lm[1]
	mouth_avg = (lm[3] + lm[4]) * 0.5

	eye_avg = (eye_left + eye_right) * 0.5
	eye_to_eye = eye_right - eye_left
	eye_to_mouth = mouth_avg - eye_avg

	x = eye_to_eye - numpy.flipud(eye_to_mouth) * [-1, 1]
	x /= numpy.hypot(*x)
	rect_scale = 1
	x *= max(numpy.hypot(*eye_to_eye) * 2.0 * rect_scale, numpy.hypot(*eye_to_mouth) * 1.8 * rect_scale)
	y = numpy.flipud(x) * [-1, 1]
	c = eye_avg + eye_to_mouth * 0.1
	quad = numpy.stack([c - x - y, c - x + y, c + x + y, c + x - y])
	qsize = numpy.hypot(*x) * 2

	quad_ori = numpy.copy(quad)
	shrink = int(numpy.floor(qsize / image_size * 0.5))
	if shrink > 1:
		h, w = img.shape[0:2]
		rsize = (int(numpy.rint(float(w) / shrink)), int(numpy.rint(float(h) / shrink)))
		img = cv2.resize(img, rsize, interpolation=cv2.INTER_AREA)
		quad /= shrink
		qsize /= shrink

	h, w = img.shape[0:2]
	border = max(int(numpy.rint(qsize * 0.1)), 3)
	crop = (int(numpy.floor(min(quad[:, 0]))), int(numpy.floor(min(quad[:, 1]))), int(numpy.ceil(max(quad[:, 0]))),
			int(numpy.ceil(max(quad[:, 1]))))
	crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, w), min(crop[3] + border, h))
	if crop[2] - crop[0] < w or crop[3] - crop[1] < h:
		img = img[crop[1]:crop[3], crop[0]:crop[2], :]
		quad -= crop[0:2]

	h, w = img.shape[0:2]
	pad = (int(numpy.floor(min(quad[:, 0]))), int(numpy.floor(min(quad[:, 1]))), int(numpy.ceil(max(quad[:, 0]))),
		   int(numpy.ceil(max(quad[:, 1]))))
	pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - w + border, 0), max(pad[3] - h + border, 0))
	if enable_padding and max(pad) > border - 4:
		pad = numpy.maximum(pad, int(numpy.rint(qsize * 0.3)))
		img = numpy.pad(img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
		h, w = img.shape[0:2]
		y, x, _ = numpy.ogrid[:h, :w, :1]
		mask = numpy.maximum(1.0 - numpy.minimum(numpy.float32(x) / pad[0],
										   numpy.float32(w - 1 - x) / pad[2]),
						  1.0 - numpy.minimum(numpy.float32(y) / pad[1],
										   numpy.float32(h - 1 - y) / pad[3]))
		blur = int(qsize * 0.02)
		if blur % 2 == 0:
			blur += 1
		blur_img = cv2.boxFilter(img, 0, ksize=(blur, blur))

		img = img.astype('float32')
		img += (blur_img - img) * numpy.clip(mask * 3.0 + 1.0, 0.0, 1.0)
		img += (numpy.median(img, axis=(0, 1)) - img) * numpy.clip(mask, 0.0, 1.0)
		img = numpy.clip(img, 0, 255)  # float32, [0, 255]
		quad += pad[:2]

	dst_h, dst_w = image_size, image_size
	template = numpy.array([[0, 0], [0, dst_h], [dst_w, dst_h], [dst_w, 0]])
	affine_matrix = cv2.estimateAffinePartial2D(quad, template, method=cv2.LMEDS)[0]
	cropped_face = cv2.warpAffine(img, affine_matrix, (dst_w, dst_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray
	affine_matrix = cv2.estimateAffinePartial2D(quad_ori, numpy.array([[0, 0], [0, image_size], [dst_w, dst_h], [dst_w, 0]]), method=cv2.LMEDS)[0]

	return cropped_face, affine_matrix


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
	face_enhancer = get_frame_processor()
	face_size = 512
	temp_face, matrix = norm_crop2(temp_frame, target_face['kps'], face_size)
	temp_face = temp_face.astype(numpy.float32)[:,:,::-1] / 255.0
	temp_face = (temp_face - 0.5) / 0.5
	temp_face = numpy.expand_dims(temp_face.transpose(2, 0, 1), axis = 0).astype(numpy.float32)

	with THREAD_SEMAPHORE:
		temp_face = face_enhancer.run(None, {face_enhancer.get_inputs()[0].name: temp_face})[0][0]

	temp_face = numpy.clip(temp_face, -1, 1)
	temp_face = (temp_face + 1) / 2
	temp_face = temp_face.transpose(1, 2, 0)
	temp_face = (temp_face * 255.0).round()
	temp_face = temp_face.astype(numpy.uint8)[:,:,::-1]

	inverse_affine = cv2.invertAffineTransform(matrix)
	h, w = temp_frame.shape[0:2]
	face_h, face_w = temp_face.shape[0:2]
	inv_restored = cv2.warpAffine(temp_face, inverse_affine, (w, h))
	mask = numpy.ones((face_h, face_w, 3), dtype = numpy.float32)
	inv_mask = cv2.warpAffine(mask, inverse_affine, (w, h))
	inv_mask_erosion = cv2.erode(inv_mask, numpy.ones((2, 2), numpy.uint8))
	inv_restored_remove_border = inv_mask_erosion * inv_restored
	total_face_area = numpy.sum(inv_mask_erosion) // 3
	w_edge = int(total_face_area ** 0.5) // 20
	erosion_radius = w_edge * 2
	inv_mask_center = cv2.erode(inv_mask_erosion, numpy.ones((erosion_radius, erosion_radius), numpy.uint8))
	blur_size = w_edge * 2
	inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
	temp_frame = inv_soft_mask * inv_restored_remove_border + (1 - inv_soft_mask) * temp_frame
	temp_frame = temp_frame.clip(0, 255).astype('uint8')
	return temp_frame


def process_frame(source_face : Face, reference_face : Face, temp_frame : Frame) -> Frame:
	many_faces = get_many_faces(temp_frame)
	if many_faces:
		for target_face in many_faces:
			temp_frame = enhance_face(target_face, temp_frame)
	return temp_frame


def process_frames(source_path : str, temp_frame_paths : List[str], update: Callable[[], None]) -> None:
	for temp_frame_path in temp_frame_paths:
		temp_frame = cv2.imread(temp_frame_path)
		result_frame = process_frame(None, None, temp_frame)
		cv2.imwrite(temp_frame_path, result_frame)
		if update:
			update()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	target_frame = cv2.imread(target_path)
	result_frame = process_frame(None, None, target_frame)
	cv2.imwrite(output_path, result_frame)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	facefusion.processors.frame.core.process_video(None, temp_frame_paths, process_frames)
