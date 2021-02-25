from gulpio import GulpDirectory
from pathlib import Path
from moviepy.editor import *
import numpy as np
import os, sys
import torch
import csv

def get_class_labels(noun_class_path, verb_class_path):
	# Get noun classes
	noun_class_list = []
	with open(noun_class_path) as f:
		for line in csv.reader(f):
			noun_class_list.append(line)
	noun_class_list = np.asarray(noun_class_list)
	noun_list = noun_class_list[1:, 1]

	# Get verb classes
	verb_class_list = []
	with open(verb_class_path) as f:
		for line in csv.reader(f):
			verb_class_list.append(line)
	verb_class_list = np.asarray(verb_class_list)
	verb_list = verb_class_list[1:, 1]

	return (noun_list, verb_list)

def load_results(results_path):
	# Load results
	results = torch.load(results_path, map_location=torch.device("cpu"))
	if isinstance(results, list):
		new_results = dict()
		first_item = results[0]
		for key in first_item.keys():
			new_results[key] = np.array([r[key] for r in results])

	return new_results

def display_rgb(rgb_frames, fps=50):
	return ImageSequenceClip(rgb_frames, fps=fps)

def display_flow(flow_frames, fps=50):
	u_frames = flow_frames[::2]
	v_frames = flow_frames[1::2]

	def flow_to_clip(flow):
		# Convert optical flow magnitude to greyscale RGB
		return ImageSequenceClip(list(np.stack([flow] * 3, axis=-1)), fps=fps)

	u_clip = flow_to_clip(u_frames)
	v_clip = flow_to_clip(v_frames) 
	return clips_array([[u_clip, v_clip]])

def main():
	# Steps
	step = 10
	if len(sys.argv) > 1: step = int(sys.argv[1])
	
	# Output directory
	out_dir = 'output_vids'
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Gets/creates count file
	cwd = Path(__file__).parent.absolute()
	count_fpath = os.path.join(cwd, 'count_file')
	if not os.path.exists(count_fpath):
		with open(count_fpath, 'w') as f:
			f.write('0')
	with open(count_fpath) as f:
		line = f.readlines()[0]
		count = int(line)
	print(f'\ncount: {count}\n')

	# Gulp dir paths
	gulp_rgb_train = '/data/epic_data/gulp/rgb_train'
	gulp_rgb_val = '/data/epic_data/gulp/rgb_validation'
	gulp_flow_train = '/data/epic_data/gulp/flow_train'
	gulp_flow_val = '/data/epic_data/gulp/flow_validation'

	# Gulp stuff
	gulp_root = Path.home()
	rgb_val = GulpDirectory(gulp_rgb_val)
	flow_val = GulpDirectory(gulp_flow_val)
	# rgb_train = GulpDirectory(gulp_rgb_train)
	# flow_train = GulpDirectory(gulp_flow_train)

	# Results and GT stuff
	results_path = '/home/smc/GIT/C1-Action-Recognition-TSN-TRN-TSM/output/trn_rgb-val.pt'
	label_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_validation.pkl'
	noun_class_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
	verb_class_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'

	# Load labels
	noun_list, verb_list = get_class_labels(noun_class_path, verb_class_path)

	# Load results
	results = load_results(results_path)
	verb_output = results['verb_output']
	noun_output = results['noun_output']
	print(f'verb_output info: {verb_output.shape}')
	print(f'noun_output info: {noun_output.shape}')

	# Creates clips
	clip_list = sorted([key for key in rgb_val.chunk_lookup.keys()])
	print(f'\nlen clip_list: {len(clip_list)}')

	# How long to run?
	limit = 5
	use_limit = False
	start = count
	stop = start + step
	if stop > len(clip_list): stop = len(clip_list)

	# Goes through all clips
	for clip_id in clip_list[start:stop]:
		print(f'\nclip: {clip_id}...')
		rgb_frames, rgb_meta = rgb_val[clip_id]
		flow_frames, flow_meta = flow_val[clip_id]

		print()
		print(f'rgb_frames type: {type(rgb_frames[0])}')
		print(f'rgb_frames len: {len(rgb_frames)}')
		print(f'rgb_frames shape: {rgb_frames[0].shape}')
		print()

		# Gets number of seconds in clip
		start_frame = int(rgb_meta['start_frame'])
		stop_frame = int(rgb_meta['stop_frame'])
		secs = int((stop_frame - start_frame) // 50)

		# Gets top verbs and nouns
		idx_verb = np.argmax(verb_output[count])
		idx_noun = np.argmax(noun_output[count])
		top_verb = verb_list[idx_verb]
		top_noun = noun_list[idx_noun]
		inf_narration = f'{top_verb} {top_noun}'

		# Create text for video
		narration = rgb_meta['narration']
		txt_clip_gt = TextClip(narration, fontsize = 60, color = 'green')
		txt_clip_gt = txt_clip_gt.set_position(('right', 'bottom')).set_duration(secs)

		txt_clip_inf = TextClip(inf_narration, fontsize = 60, color = 'red')
		txt_clip_inf = txt_clip_inf.set_position(('left', 'bottom')).set_duration(secs)

		# Create clips and videos
		clip = clips_array([[display_rgb(rgb_frames), display_flow(flow_frames)]])
		video = CompositeVideoClip([clip, txt_clip_gt, txt_clip_inf])

		# Writes video
		video.write_videofile(os.path.join(out_dir, f'{clip_id}.mp4'))
		
		# Prints GT
		print(f'GT rgb_meta: {narration}')
		print(f'inferred verb, noun: {top_verb}, {top_noun}')
		print(f'INFER: {top_noun}, {top_verb}')

		# Closes clip, txt, and video
		txt_clip_gt.close()
		txt_clip_inf.close()
		clip.close()
		video.close()
		
		# Updates count and writes count file
		count +=1 
		count_file = open(count_fpath, 'w')
		count_file.write(f'{count}')
		count_file.close()
		print(f'segment count: {count}')
		print('Done.')
		if use_limit and count >= limit:
			break

	# Prints stuff
	print(f'\nCount of clips written: {count}')

if __name__ == '__main__':
	main()