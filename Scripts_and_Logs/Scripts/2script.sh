#!/bin/bash
#script to remove all extra frames created from gif other than the first one i.e. the png file
echo 'input directory'
read direc

types='.p01';
files=($(ls $direc | grep $types));
count=1;

for x in ${files[@]}; do
	echo 'removing unwanted png files created from gifs' $direc/$x;
	y=${x:0:-4}; #check if multiple png frames like .p01, .p02.. are created from gif?
	echo $y;	
	files2remove=($(ls $direc | grep $y));
	for i in ${files2remove[@]}; do
		fileExt=${i: -4}; #check the extension of file if it is png donot remove it
		echo "Extension of file " $fileExt;
		if [[ "$fileExt" == ".png" ]];  
		then
			echo "PNG FILE SPOTTED. NOT REMOVING IT";
		else
			echo "REMOVING FILE " $direc/$i;
			rm $direc/$i;
			((count+=1));
		fi
	done
	
done
echo "#Files removed " $count;
