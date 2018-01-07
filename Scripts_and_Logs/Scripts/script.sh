#!/bin/bash

echo 'input directory'
read direc
types='.jpeg\|.gif\|.GIF\|.png\|.jpg\|.JPG\|.PNG\|.JPEG';
files=($(ls $direc | grep -v $types));
count=1;

for x in ${files[@]}; do
	echo 'removing file ' $direc/$x;
	rm $direc/$x;
	((count+=1));
done
echo $count;
