#!/bin/bash

i='123dsf233432.png';
fileExt=${i: -4}; 
echo "Extension of file" $fileExt;
if [[ "$fileExt" = ".png" ]];  
then
		echo "PNG FILE SPOTTED. NOT REMOVING IT";
else
		echo "REMOVING FILE ";
fi		

