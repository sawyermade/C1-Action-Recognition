import csv, json

def extract_class_and_label(fpath):
	temp_list = []
	with open(fpath) as cf:
		for line in csv.reader(cf):
			temp_list.append(line)

	# Pull info
	temp_list = temp_list[1:]
	class_list, label_list = [], []
	for cl in temp_list:
		c = int(cl[0])
		l = cl[1]
		class_list.append(c)
		label_list.append(l)

	return (class_list, label_list)

def get_tail_classes(fpath):
	class_list = []
	with open(fpath) as cf:
		for line in csv.reader(cf):
			class_list.append(line)

	class_list = [int(c[0]) for c in class_list[1:]]
	return class_list

def reorder_classes(class_list, label_list, tail_classes):
	class_notail, label_notail = [], []
	for c, l in zip(class_list, label_list):
		if c not in tail_classes:
			class_notail.append(c)
			label_notail.append(l)

	class_notail = [[c, i] for i, c in enumerate(class_notail)]
	class_notail = [c + [l] for c, l in zip(class_notail, label_notail)]
	tail_num = len(label_notail)
	class_notail.append(['tails', tail_num, 'tails'])
	label_notail.append('tails')

	class_dict = {}
	for cl in class_notail:
		class_dict.update({
			cl[0] : {
				'class' : cl[1],
				'label' : cl[2]
			}
		})

	return class_notail, label_notail, class_dict

def main():
	# Results and GT paths
	noun_class_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv'
	verb_class_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv'
	tail_noun_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_tail_nouns.csv'
	tail_verb_path = '/home/smc/GIT/epic-kitchens-100-annotations/EPIC_100_tail_verbs.csv'

	# Get classes and labels
	noun_class_list_full, noun_label_list_full = extract_class_and_label(noun_class_path)
	verb_class_list_full, verb_label_list_full = extract_class_and_label(verb_class_path)

	# Get tails
	tail_noun_classes = get_tail_classes(tail_noun_path)
	tail_verb_classes = get_tail_classes(tail_verb_path)

	# Reorder
	noun_class_list_notail, noun_label_list_notail, noun_dict = reorder_classes(noun_class_list_full, noun_label_list_full, tail_noun_classes)
	verb_class_list_notail, verb_label_list_notail, verb_dict = reorder_classes(verb_class_list_full, verb_label_list_full, tail_verb_classes)

	# Write json
	with open('noun_classes_notail.json', 'w') as jf:
		json.dump(noun_class_list_notail, jf)

	with open('verb_classes_notail.json', 'w') as jf:
		json.dump(verb_class_list_notail, jf)

	with open('noun_labels_notail.json', 'w') as jf:
		json.dump(noun_label_list_notail, jf)

	with open('verb_labels_notail.json', 'w') as jf:
		json.dump(verb_label_list_notail, jf)

	with open('noun_dict_notail.json', 'w') as jf:
		json.dump(noun_dict, jf, indent=3)

	with open('verb_dict_notail.json', 'w') as jf:
		json.dump(verb_dict, jf, indent=3)

if __name__ == '__main__':
	main()