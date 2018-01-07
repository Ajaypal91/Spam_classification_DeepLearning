#!/bin/bash
#script to remove corrupted gif files
echo 'input directory'
read direc

types='.gif\|.GIF';
files=($(ls $direc | grep $types));
count=1;

for x in ${files[@]}; do
	echo 'removing unwanted gif files ' $direc/$x;
	rm $direc/$x;
	((count+=1));
	
	
done
echo "#Files removed " $count;
