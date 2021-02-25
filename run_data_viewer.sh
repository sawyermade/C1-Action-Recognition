#!/bin/bash
typeset -i count=$(cat count_file)
count_max=9669

while [ $count -lt $count_max ]
do
	python data_viewer.py 100
	typeset -i count=$(cat count_file)
done
