#!/bin/bash
python evaluate.py \
	/home/smc/GIT/C1-Action-Recognition-TSN-TRN-TSM/output/trn_rgb-val.pt \
	/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_validation.pkl \
	--tail-verb-classes-csv /home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_tail_verbs.csv \
	--tail-noun-classes-csv /home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_tail_nouns.csv \
	--unseen-participant-ids /home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_unseen_participant_ids_validation.csv