#!/bin/bash

FILES=../../data/alltmy3a/*
output=../../data/renamed
counter=1
for f in $FILES
do
  echo "Copying $f ... to $output/tmy.$counter"
  cp $f $output/tmy.$counter
  ((counter++))
  # take action on each file. $f store current file name
done
