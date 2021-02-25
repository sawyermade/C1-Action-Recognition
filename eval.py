import pandas as pd
import numpy as np
import torch
import yaml

from utils.actions import action_id_from_verb_noun
from utils.metrics import compute_metrics
# from utils.results import load_results
from utils.scoring import compute_action_scores

def main():
	# Args
	ckpt_path = '/home/smc/GIT/C1-Action-Recognition-TSN-TRN-TSM/models/trn_rgb.ckpt'
	results_path = '/home/smc/GIT/C1-Action-Recognition-TSN-TRN-TSM/output/trn_rgb-val.pt'
	label_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_validation.pkl'
	tail_verbs_path = 'annotations/EPIC_100_tail_verbs.csv'
	tail_nouns_path = 'annotations/EPIC_100_tail_nouns.csv'
	unseen_path = 'annotaions/EPIC_100_unseen_participant_ids.csv'

	# Load results
	results = torch.load(results_path, map_location=torch.device("cpu"))
	if isinstance(results, list):
		new_results = dict()
		first_item = results[0]
		for key in first_item.keys():
			new_results[key] = np.array([r[key] for r in results])
	results = new_results

	# Load labels

	# Test
	key = 'verb_output'
	print(f'{key} count: {results[key].shape}')
	key = 'noun_output'
	print(f'{key} count: {results[key].shape}')
	print(results.keys())
	print(results)

if __name__ == '__main__':
	main()