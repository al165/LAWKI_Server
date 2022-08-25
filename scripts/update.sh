#!/bin/bash

echo "Checking for updates for LAWKI_Server"
cd /home/lawki/LAWKI_Server/
git pull

sleep 1

echo " - Done"
echo "Checking for updates for LAWKI_Player"

cd /home/lawki/openFrameworks/apps/myApps/lawki_player/
git pull

sleep 1
echo " - Done"
sleep 5

