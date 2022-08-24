
jack_connect pure_data_0:output0 S12:playback_1
jack_connect pure_data_0:output1 S12:playback_2

jack_connect pure_data_0:output2 S12:playback_1
jack_connect pure_data_0:output3 S12:playback_2

#jack_connect pure_data_0:output2 system:playback_1
#jack_connect pure_data_0:output3 system:playback_2

jack_connect pure_data_0:output4 S34:playback_1
jack_connect pure_data_0:output5 S34:playback_2

jack_connect pure_data_0:output6 S34:playback_1
jack_connect pure_data_0:output7 S34:playback_2

jack_connect "PulseAudio JACK Sink":front-left pure_data_0:input0
jack_connect "PulseAudio JACK Sink":front-right pure_data_0:input1
jack_connect "PulseAudio JACK Sink-01":front-left pure_data_0:input0
jack_connect "PulseAudio JACK Sink-01":front-right pure_data_0:input1
