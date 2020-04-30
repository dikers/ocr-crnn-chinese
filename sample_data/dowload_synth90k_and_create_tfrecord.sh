#!/bin/sh

# download synth90k
wget -c http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
tar -xzvf mjsynth.tar.gz

# create synth90k image_list.txt
find ./mnt/ | xargs ls -d | grep jpg > image_list_all.txt





