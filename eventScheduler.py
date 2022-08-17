import os
import sys
from time import time
from queue import Empty
from threading import Thread, Event, Timer
from multiprocessing import Process

from logprint import logprint


def sendSample(*args, **kwargs):
    '''
    - args:  (addr, msg)
        expect addr to be of form /station_number/type/dest
        where type in {'video', 'audio', 'filter'}
    - kwargs: 'clients'
    '''

    addr, msg = args
    clients = kwargs['clients']
    #logprint('sendSample:', addr, msg)

    # get station number from addr
    #station_number = int(addr.split('/')[1][8:])
    station_number = 1

    try:
        clients[station_number].send_message(addr, msg)
        #logprint(addr, msg)
    except KeyError:
        logprint(f'could not send to station {station_number}, not enough clients')




class EventScheduler(Thread):
    '''
    EventScheduler is used by any process that creates timed events
    '''

    def __init__(self, msg_queue, send_function=print, send_function_kwargs={}):

        super(EventScheduler, self).__init__()

        self.msg_queue = msg_queue
        self.send_function = send_function
        self.send_function_kwargs = send_function_kwargs

        self.timers = []
        self.events = None
        self.start_time = time()
        self.stop = Event()

        logprint('init')


    def run(self):
        '''Start thread'''

        while not self.stop.is_set():
            self.checkMessages()


    def join(self, timeout=1):
        '''Close thread'''
        logprint('joined')
        self.cancelTimers()
        self.stop.set()
        super(EventScheduler, self).join(timeout)


    def cancelTimers(self):
        '''Stop all timers'''
        for t in self.timers:
            t.cancel()
        self.timers = []


    def addTimers(self, events):
        '''Create and start timers for a list of events

        Params
        ------
        events : iterable
            Collection of events, where each event is (start_time, msg)
        '''

        if len(events) == 2 and isinstance(events[0], float):
            # events is only on event and not an iterable
            events = [event]

        now = time()
        for event in events:
            wait = event[0] - now
            if wait < 0.0:
                print('recieved event from past', wait, event)
                wait = 0.0
                #continue
            msg = event[1]
            timer = Timer(wait, self.send_function, args=msg, kwargs=self.send_function_kwargs)
            timer.start()
            self.timers.append(timer)


    def checkMessages(self):
        '''Check for updates

        Message types: 'events', 'end'
        '''
        try:
            msgs = self.msg_queue.get(block=True, timeout=1.0)
            if msgs is not None:
                #logprint('new message')
                if msgs['type'] == 'events':
                    es = msgs['events']
                    self.addTimers(es)
                elif msgs['type'] == 'end':
                    self.join()

        except Empty:
           pass
