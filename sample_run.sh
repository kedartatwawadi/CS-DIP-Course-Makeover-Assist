#!/bin/bash

cd src
make
cd ..

for i in Data/sample/*.jpg; do
./src/makeover $i Data/glass.jpg
done



