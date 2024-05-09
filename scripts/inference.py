#!/usr/bin/env python3

#
# Copyright (c) Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import argparse
import logging
import os
import sys
import time
import traceback

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
from gi.repository import GLib, GObject, Gst

from gstreamer import utils

from PIL import Image


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logging.info('End-of-stream')
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logging.warn('%s: %s' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logging.error('%s: %s' % (err, debug))
        loop.quit()
    return True


def add_probe(pipeline, element_name, callback, pad_name="sink", probe_type=Gst.PadProbeType.BUFFER):
    logging.info("add_probe: Adding probe to %s pad of %s" % (pad_name, element_name))
    element = pipeline.get_by_name(element_name)
    if not element:
        raise Exception("Unable to get element %s" % element_name)
    sinkpad = element.get_static_pad(pad_name)
    if not sinkpad:
        raise Exception("Unable to get %s pad of %s" % (pad_name, element_name))
    sinkpad.add_probe(probe_type, callback, 0)


def format_clock_time(ns):
    """Format time in nanoseconds like 01:45:35.975000000"""
    s, ns = divmod(ns, 1000000000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "%u:%02u:%02u.%09u" % (h, m, s, ns)


def set_event_message_meta_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if gst_buffer:
        caps = Gst.Caps.new_empty_simple("pravega-stream-metadata")
        meta = gst_buffer.get_reference_timestamp_meta(caps)

        if not gst_buffer.has_flags(Gst.BufferFlags.DELTA_UNIT) and meta.duration != Gst.CLOCK_TIME_NONE:
            logging.info("set_event_message_meta_probe: %s:%s: pts=%23s, dts=%23s, offset=%d, event_head_offset=%d, event_tail_offset=%d, duration=%23s, size=%8d" % (
                pad.get_parent_element().name,
                pad.name,
                format_clock_time(gst_buffer.pts),
                format_clock_time(gst_buffer.dts),
                gst_buffer.offset,
                meta.timestamp,
                meta.duration,
                format_clock_time(gst_buffer.duration),
                gst_buffer.get_size()
            ))
            # To do inference here
            caps = pad.get_current_caps()
            image_array = utils.gst_buffer_with_caps_to_ndarray(gst_buffer, caps)
            image = Image.fromarray(image_array)
            image.save('image.jpg')
    return Gst.PadProbeReturn.OK

def main():
    parser = argparse.ArgumentParser(description='Pravega inferene job')
    parser.add_argument('--controller', default='172.28.1.1:9090')
    parser.add_argument('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add_argument('--scope', default='examples')
    parser.add_argument('--stream', default='urv6')
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.info('args=%s' % str(args))

    # Set GStreamer log level.
    # if not 'GST_DEBUG' in os.environ:
    #     os.environ['GST_DEBUG'] = 'pravegasrc:DEBUG'

    # Standard GStreamer initialization.
    Gst.init(None)
    logging.info(Gst.version_string())

    pipeline_description = (
        'pravegasrc name=src ! '
        'decodebin !'
        'videoconvert !'
        'video/x-raw,format=RGB,width=960,height=540,framerate=15/1 ! '
        'fakesink name=sink'
    )
    logging.info('Creating pipeline: ' +  pipeline_description)
    pipeline = Gst.parse_launch(pipeline_description)

    pravegasrc = pipeline.get_by_name('src')
    pravegasrc.set_property('controller', args.controller)
    pravegasrc.set_property('stream', '%s/%s' % (args.scope, args.stream))
    pravegasrc.set_property("start-mode", 'earliest')
    pravegasrc.set_property("end-mode", 'latest')
    pravegasrc.set_property("allow-create-scope", True)

    sink = pipeline.get_by_name("sink")
    if sink:
        add_probe(pipeline, "sink", set_event_message_meta_probe, pad_name='sink')

    # Create an event loop and feed GStreamer bus messages to it.
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    # Start play back and listen to events.
    logging.info('Starting pipeline')
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        logging.error(traceback.format_exc())
        # Cleanup GStreamer elements.
        pipeline.set_state(Gst.State.NULL)
        raise

    pipeline.set_state(Gst.State.NULL)
    logging.info('END')


if __name__ == '__main__':
    main()
