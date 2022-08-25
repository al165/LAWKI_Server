#!/bin/bash

echo "~~ STARTING LAWKI-PASSAGES ~~" > start.log
cd /home/lawki/LAWKI_Server/
echo "" >> start.log

whoami >> start.log
date >> start.log
pwd >> start.log

killall -s TERM python
killall -s TERM purr-data

sleep 3

echo " - setting up default audio files..." >> start.log
cp music_generator/default_files/* /tmp/

echo " - setting up sound..." >> start.log
# make sure JACK is running

# start Pd and run the LAWKI patch
purr-data -jack -rt -inchannels 2 -outchannels 8 music_player/video_sound.pd > /dev/null &

echo " - moving windows out the way" >> start.log
sleep 3
wmctrl -r purr-data -t 1
wmctrl -r video_sound.pd -t 1
wmctrl -r teamviewer -t 1

echo " - (waiting for sound cards to be free..."
sleep 7

# rename ALSA devices
scripts/make_alsa_outs.sh >> start.log

# disconnect all of Pd's connections
scripts/disconnect_all.sh >> start.log

# connect Pd to correct outputs
scripts/make_jack_connections.sh >> start.log

echo " - sound done" >> start.log

echo " - starting video server" >> start.log
python video_server.py &
echo " - done" >> start.log


echo " - starting loop generator and sample collector" >> start.log
python loop_generator.py &
python sample_collector.py &
echo " - done" >> start.log

echo " - starting the LAWKI player" >> start.log
../openFrameworks/apps/myApps/lawki_player/bin/lawki_player &

#python /home/lawki/development/lawki_player.py -l


echo " - DONE " >> start.log
