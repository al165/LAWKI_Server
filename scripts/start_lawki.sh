#!/bin/bash

echo "~~ STARTING LAWKI-PASSAGES ~~" > start.log
cd /home/lawki
echo "" >> start.log

whoami >> start.log
date >> start.log
pwd >> start.log

killall python
killall purr-data

sleep 1

echo " - setting up sound..." >> start.log
# make sure JACK is running

# rename ALSA devices
/home/lawki/LAWKI_scripts/make_alsa_outs.sh >> start.log

# start Pd and run the LAWKI patch
purr-data -jack -rt -inchannels 2 -outchannels 8 /home/lawki/LAWKI_Server/music_player/video_sound.pd > /dev/null &

sleep 2

# disconnect all of Pd's connections
/home/lawki/LAWKI_Server/scripts/disconect_all.sh >> start.log

sleep 1

# connect Pd to correct outputs
/home/lawki/LAWKI_Server/scripts/make_jack_connections.sh >> start.log

echo " - sound done" >> start.log

echo " - starting video server" >> start.log
sleep 1
python /home/lawki/LAWKI_Server/video_server.py &
echo " - done" >> start.log


echo " - starting loop generator and sample collector" >> start.log
python /home/lawki/LAWKI_Server/loop_generator.py &
python /home/lawki/LAWKI_Server/sample_collector.py &
echo " - done" >> start.log

echo " - starting the LAWKI player" >> start.log
sleep 2

/home/lawki/openFrameworks/apps/myApps/lawki_player/bin/lawki_player &
#python /home/lawki/development/lawki_player.py -l


echo " - DONE " >> start.log
