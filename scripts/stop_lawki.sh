#!/bin/bash

echo "~~ STOPPING LAWKI-PASSAGES ~~" >> start.log
date >> start.log

/home/lawki/LAWKI_Server/scripts/disconnect_all.sh >> start.log

killall -s TERM python
killall -s TERM purr-data

killall -s TERM lawki_player

echo "~~ ENDED ~~" >> start.log

