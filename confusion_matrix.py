import numpy as np, torch, os, sys, csv, json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools

def load_results(results_path):
	# Load results
	results = torch.load(results_path, map_location=torch.device("cpu"))
	if isinstance(results, list):
		new_results = dict()
		first_item = results[0]
		for key in first_item.keys():
			new_results[key] = np.array([r[key] for r in results])

	return new_results

def get_class_labels(noun_class_path, verb_class_path):
	# Get noun classes
	noun_class_list = []
	with open(noun_class_path) as f:
		for line in csv.reader(f):
			noun_class_list.append(line)
	noun_class_list = np.asarray(noun_class_list)
	noun_labels = noun_class_list[1:,1]

	# Get verb classes
	verb_class_list = []
	with open(verb_class_path) as f:
		for line in csv.reader(f):
			verb_class_list.append(line)
	verb_class_list = np.asarray(verb_class_list)
	verb_labels = verb_class_list[1:,1]

	return (noun_labels, verb_labels)

def get_gt_classes(gt_path):
	# Read CSV
	gt_full_list = []
	with open(gt_path) as f:
		for line in csv.reader(f):
			gt_full_list.append(line)
	gt_full_list = np.asarray(gt_full_list)

	# Parse for verb and noun classes
	gt_verbs = np.asarray([int(f) for f in gt_full_list[1:, 10]])
	gt_nouns = np.asarray([int(f) for f in gt_full_list[1:, 12]])
	print(f'gt_verbs: {gt_verbs.shape}')
	print(f'gt_nouns: {gt_nouns.shape}')

	return gt_verbs, gt_nouns

def get_top_n(vn_list, topn=1):
	top_list = []
	for vn in vn_list:
		if topn == 1:
			idx = np.argmax(vn)
		else:
			idx = np.argsort(vn)[::-1][:topn]
		top_list.append(idx)
	
	top_list = np.asarray(top_list)
	return top_list

def print_results_v_gt(inf_list, gt_list, label_list, topn=1, fpath=None):
	# print('Ground Truth, Inferred/Inferred_List')
	comp_dict = {}
	for i in range(inf_list.shape[0]):
		# Gets gt and inf results
		gt  = label_list[gt_list[i]]

		# Add idx to dict
		if topn == 1:
			if i not in comp_dict.keys():
				comp_dict.update({
					i : {
						'gt'  : '-1',
						'inf' : '-1'
					} 
				})
			inf = label_list[inf_list[i]]
			comp_dict[i]['inf'] = inf
		else:
			if i not in comp_dict.keys():
				comp_dict.update({
					i : {
						'gt'  : '-1',
						'inf' : []
					} 
				})
			inf = [label_list[f] for f in inf_list[i]]
			comp_dict[i]['inf'] = inf

		# Adds gt to dict
		comp_dict[i]['gt'] = gt
		
		# Prints
		# print(f'{gt}, {inf}')

	# Saves dict to json
	if fpath is not None:
		with open(fpath, 'w') as f:
			json.dump(comp_dict, f, indent=3)

def convert_class2label(class_list, gt_list, label_list, topn=1):
	labels = []
	gt_labels = []
	for i in range(class_list.shape[0]):
		cgt = gt_list[i]
		gt_labels.append(label_list[cgt])
		if topn == 1:
			c = class_list[i]
			
		else:
			if cgt in class_list[i]:
				c = cgt
			else:
				c = class_list[i, 0]

		l = label_list[c]
		labels.append(l)

	labels = np.asarray(labels)
	gt_labels = np.asarray(gt_labels)
	return (labels, gt_labels)

def remove_tails(preds, gts, tails, dict_notail):
	# Convert to python lists
	preds = preds.tolist()
	gts = gts.tolist()
	tails = tails.tolist()
	# print(preds[0], gts[0], tails[0])

	new_preds, new_gts = [], []
	for pred, gt in zip(preds, gts):
		# if pred not in tails and gt not in tails:
		# if gt not in tails or pred not in tails:
		# 	new_preds.append(pred)
		# 	new_gts.append(gt)
		if pred in tails:
			new_pred = int(dict_notail['tails']['class'])
		else:
			new_pred = int(dict_notail[str(pred)]['class'])

		if gt in tails:
			new_gt = int(dict_notail['tails']['class'])
		else:
			new_gt = int(dict_notail[str(gt)]['class'])

		new_preds.append(new_pred)
		new_gts.append(new_gt)

	return (np.asarray(new_preds), np.asarray(new_gts))

def topn_to_1(class_list, gt_list):
	classes = []
	for i in range(class_list.shape[0]):
		if gt_list[i] in class_list[i]:
			c = gt_list[i]
		else:
			c = class_list[i, 0]
		classes.append(c)

	classes = np.asarray(classes)
	return classes

def plot_confusion_matrix_mine(cm, classes, fname, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(10,10)):

	plt.figure(figsize=figsize)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(
			j, i, cm[i, j],
			horizontalalignment="center",
			color="white" if cm[i, j] > thresh else "black"
		)

	# plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	# plt.figure(figsize=(10, 10))

	plt.savefig(fname)
	# plt.show()

def create_noun_cm(gts, preds, cm_shape):
	cm = np.zeros((cm_shape, cm_shape), dtype=np.int)
	print(f'\ncm shape: {cm.shape}\n')
	for i in range(gts.shape[0]):
		gt = gts[i]
		pred = preds[i]
		cm[gt, pred] += 1

	return cm

def get_verb_freq(gts, num_classes):
	verb_freq = np.zeros((num_classes), dtype=np.int)
	for gt in gts:
		verb_freq[gt] += 1

	return verb_freq

def get_verb_tail_freq(gts, labels, tail_classes):
	verb_tail_dict = {}
	for gt in gts:
		if gt in tail_classes:
			l = labels[gt]
			if l not in verb_tail_dict.keys():
				verb_tail_dict.update({l : 1})
			else:
				verb_tail_dict[l] += 1

	verb_tail_dict = {key: v for key, v in sorted(verb_tail_dict.items(), reverse=True, key=lambda item: item[1])}
	return verb_tail_dict

def cm_to_csv(fpath, cm, labels):
	with open(fpath, 'w') as cf:
		writer = csv.writer(cf)
		writer.writerow(labels)
		for row in cm:
			writer.writerow(row)

def main():
	# Results and GT paths
	results_path = '/home/smc/GIT/C1-Action-Recognition-TSN-TRN-TSM/output/trn_rgb-val.pt'
	gt_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_validation.csv'
	noun_class_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
	verb_class_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
	tail_noun_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_tail_nouns.csv'
	tail_verb_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_tail_verbs.csv'

	# Load results
	results = load_results(results_path)
	verb_output = results['verb_output']
	noun_output = results['noun_output']
	print(f'verb_output info: {verb_output.shape}')
	print(f'noun_output info: {noun_output.shape}')

	# Load classes
	noun_labels, verb_labels = get_class_labels(noun_class_path, verb_class_path)

	# Load GT
	gt_verbs, gt_nouns = get_gt_classes(gt_path)
	print(f'gt_verbs, gt_nouns: {gt_verbs.shape}, {gt_nouns.shape}')

	# Load tail classes
	with open(tail_noun_path) as f:
		tail_noun_classes = [line for line in f]
		tail_noun_classes = [int(c) for c in tail_noun_classes[1:]]
		tail_noun_classes = np.asarray(tail_noun_classes)

	with open(tail_verb_path) as f:
		tail_verb_classes = [line for line in f]
		tail_verb_classes = [int(c) for c in tail_verb_classes[1:]]
		tail_verb_classes = np.asarray(tail_verb_classes)

	# Get top1 and top5 verbs and nouns
	verb_top1 = get_top_n(verb_output, 1)
	verb_top5 = get_top_n(verb_output, 5)
	noun_top1 = get_top_n(noun_output, 1)
	noun_top5 = get_top_n(noun_output, 5)
	print(f'verbs top1, top5: {verb_top1.shape}, {verb_top5.shape}')

	# Print results vs gt
	print_results_v_gt(verb_top1, gt_verbs, verb_labels, 1, 'verb_top1.json')
	print_results_v_gt(verb_top5, gt_verbs, verb_labels, 5, 'verb_top5.json')
	print_results_v_gt(noun_top1, gt_nouns, noun_labels, 1, 'noun_top1.json')
	print_results_v_gt(noun_top5, gt_nouns, noun_labels, 5, 'noun_top5.json')

	# Create with labels
	# verb_top1_labels, gt_verb_labels = convert_class2label(verb_top1, gt_verbs, verb_labels, 1)
	# verb_top5_labels, _ = convert_class2label(verb_top5, gt_verbs, verb_labels, 5)
	# noun_top1_labels, gt_noun_labels = convert_class2label(noun_top1, gt_nouns, noun_labels, 1)
	# noun_top5_labels, _ = convert_class2label(noun_top5, gt_nouns, noun_labels, 5)

	# Gets score and confusion matrix
	score_verbs_top1 = metrics.accuracy_score(gt_verbs, verb_top1)
	print(f'\nverb_top1 accuracy_score: {score_verbs_top1}\n')
	# cm = confusion_matrix(gt_verbs, verb_top1)
	# plot_confusion_matrix(cm, verb_labels, 'cm_verbs.png')

	verb_top5_1 = topn_to_1(verb_top5, gt_verbs)
	score_verbs_top5 = metrics.accuracy_score(gt_verbs, verb_top5_1)
	print(f'\nverb_top5 accuracy_score: {score_verbs_top5}\n')

	score_nouns_top1 = metrics.accuracy_score(gt_nouns, noun_top1)
	print(f'\nnoun_top1 accuracy_score: {score_nouns_top1}\n')
	# cm = confusion_matrix(gt_nouns, noun_top1)
	# plot_confusion_matrix(cm, noun_labels, 'cm_nouns.png')

	noun_top5_1 = topn_to_1(noun_top5, gt_nouns)
	score_nouns_top5 = metrics.accuracy_score(gt_nouns, noun_top5_1)
	print(f'\nnoun_top5 accuracy_score: {score_nouns_top5}\n')


	print(f'gt_noun min, max class: {gt_nouns.min()}, {gt_nouns.max()}')
	print(f'gt_verb min, max class: {gt_verbs.min()}, {gt_verbs.max()}')


	### TAIL STUFF
	# No tails files
	noun_dict_notail_path = 'noun_dict_notail.json'
	verb_dict_notail_path = 'verb_dict_notail.json'
	noun_labels_notail_path = 'noun_labels_notail.json'
	verb_labels_notail_path = 'verb_labels_notail.json'

	noun_dict_notail = json.load(open(noun_dict_notail_path))
	verb_dict_notail = json.load(open(verb_dict_notail_path))
	noun_labels_notail = json.load(open(noun_labels_notail_path))
	verb_labels_notail = json.load(open(verb_labels_notail_path))
	noun_labels_notail = np.asarray(noun_labels_notail)
	verb_labels_notail = np.asarray(verb_labels_notail)
	# sys.exit()
	

	print(f'verb_top1: {verb_top1.shape}\n{verb_top1}')
	print(f'gt_verbs: {gt_verbs.shape}\n{gt_verbs}')


	notail_verb_top1, notail_gt_verbs = remove_tails(verb_top1, gt_verbs, tail_verb_classes, verb_dict_notail)
	print(f'\nnotail verb preds, notail verb gt: {len(notail_verb_top1)}, {len(notail_gt_verbs)}')
	cmv = confusion_matrix(notail_gt_verbs, notail_verb_top1)
	# plot_confusion_matrix_mine(cmv, verb_labels_notail, 'cm_verbs.png')

	notail_noun_top1, notail_gt_nouns = remove_tails(noun_top1, gt_nouns, tail_noun_classes, noun_dict_notail)
	print(f'\nnotail_noun_top1: {notail_noun_top1.shape}, {notail_gt_nouns.shape}')
	print(f'\nmax noun class: {notail_noun_top1.max()}, {notail_gt_nouns.max()}, {noun_labels_notail.shape}')
	# cmn = confusion_matrix(notail_gt_nouns, notail_noun_top1)
	cmn = create_noun_cm(notail_gt_nouns, notail_noun_top1, noun_labels_notail.shape[0])
	# plot_confusion_matrix_mine(cmn, noun_labels_notail, 'cm_nouns.png', figsize=(40, 40))

	print(f'\ncmv: shape={cmv.shape}, \n{cmv}')
	print(f'\ncmn: shape={cmn.shape}, max={cmn.max()}')

	verb_freq = get_verb_freq(notail_gt_verbs, verb_labels_notail.shape[0])
	print('\nverb freq:')
	for v, l in zip(verb_freq, verb_labels_notail):
		print(f'{l}: {v}')
	print(f'tail verb class number: {tail_verb_classes.shape[0]}')

	verb_tail_dict = get_verb_tail_freq(gt_verbs, verb_labels, tail_verb_classes)
	print('\nverb tail freq:')
	count = 0
	for l, v in verb_tail_dict.items():
		print(f'{l}: {v}')
		count += v
	print(f'\nnum classes: {len(verb_tail_dict.keys())}, {tail_verb_classes.shape[0]}')
	print(f'total: {count}')

	# Saves CMs to csv
	cm_to_csv('cm_verb_notail.csv', cmv, verb_labels_notail)
	cm_to_csv('cm_noun_notail.csv', cmn, noun_labels_notail)

if __name__ == '__main__':
	main()