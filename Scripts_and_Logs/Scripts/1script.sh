#!/bin/bash
#script to convert gif files to png and then remove the gif that were converted
echo 'input directory'
read direc

types='.gif\|.GIF';
files=($(ls $direc | grep $types));
count=1;

for x in ${files[@]}; do
	echo 'changing file from GIF to png ' $direc/$x;
	y=${x:0:-4}.png; #check if png file is created?
	gif2png -r  $direc/$x;
	echo $y;
	c=($(ls $direc | grep $y | wc -l));
	echo $c;
	if ((c > 0));
	then
		echo "PNG FILE CREATED";
		echo "Removing GIF file";
		rm $direc/$x;
		((count+=1));
	else
		echo "COULD NOT CREATE PNG";
	fi
	
done
echo $count;
