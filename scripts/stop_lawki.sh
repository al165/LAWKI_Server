#!/bin/bash

echo "~~ STOPPING LAWKI-PASSAGES ~~" >> start.log
date >> start.log

/home/lawki/LAWKI_Server/scripts/disconect_all.sh >> start.log

killall python
killall purr-data

killall lawki_player

echo "~~ ENDED ~~" >> start.log

