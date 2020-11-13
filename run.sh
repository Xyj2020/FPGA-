#£¡/bin/bash

read -p "select mode: " a
if [ $a -eq 1 ]
then
  \make clean
  \make
  ./video_analysis video/ship14.mp4 n
fi
