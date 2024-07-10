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

## Import other packages
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

## General libraries
import argparse
import logging
import time
import traceback
from datetime import datetime
import json

## Import User policies
from policies.components import get_model, inference, do_sampling
from policies.constants import (PRAVEGA_CONTROLLER, PRAVEGA_SCOPE,
                                MILVUS_HOST, MILVUS_PORT, MILVUS_NAMESPACE,
                                DO_LATENCY_LOG, LOG_PATH)

sampling_fn = do_sampling()

## Gstreamer and Pravega libraries
import gi # type: ignore
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
from gi.repository import GLib, GObject, Gst # type: ignore
from gstreamer import utils

## Milvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

## Setup metric logging file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
latency_log = None
global_var = {"counter": 0}


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


def add_probe(pipeline, element_name, callback, pad_name="sink", probe_type=Gst.PadProbeType.BUFFER, model=None, milvus=None, global_milvus=None, device=None):
    logging.info("add_probe: Adding probe to %s pad of %s" % (pad_name, element_name))
    element = pipeline.get_by_name(element_name)
    if not element:
        raise Exception("Unable to get element %s" % element_name)
    sinkpad = element.get_static_pad(pad_name)
    if not sinkpad:
        raise Exception("Unable to get %s pad of %s" % (pad_name, element_name))
    sinkpad.add_probe(probe_type, callback, {"model": model, "milvus": milvus, "global_milvus": global_milvus, "device": device})


def format_clock_time(ns):
    """Format time in nanoseconds like 01:45:35.975000000"""
    s, ns = divmod(ns, 1000000000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "%u:%02u:%02u.%09u" % (h, m, s, ns)


def set_event_message_meta_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if gst_buffer:
        # Only process keyframes
        caps = Gst.Caps.new_empty_simple("pravega-stream-metadata")
        meta = gst_buffer.get_reference_timestamp_meta(caps)
        if not gst_buffer.has_flags(Gst.BufferFlags.DELTA_UNIT) and meta.duration != Gst.CLOCK_TIME_NONE:   
            # Debugging info
            time_nanosec = time.time_ns()
            logging.info("set_event_message_meta_probe: %s:%s: pts=%23s, dts=%23s, offset=%d, event_head_offset=%d, event_tail_offset=%d, duration=%23s, size=%8d, e2e latency(ms)=%s" % (
                pad.get_parent_element().name,
                pad.name,
                format_clock_time(gst_buffer.pts),
                format_clock_time(gst_buffer.dts),
                gst_buffer.offset,
                meta.timestamp,
                meta.duration,
                format_clock_time(gst_buffer.duration),
                gst_buffer.get_size(),
                ((time_nanosec - (gst_buffer.pts - 37000000000)) / 1000000.)
            ))
            
            # Fetch the image from buffer
            caps = pad.get_current_caps()
            image_array = utils.gst_buffer_with_caps_to_ndarray(gst_buffer, caps)
            global_var["size"] = image_array.shape # Store the original image size to calculate real size
            
            # Perform inference
            start_time = time.time()
            embeds = inference(u_data["model"], image_array, u_data["device"])
            embeds = embeds.detach().numpy()
            inference_time = time.time()
            
            # Insert the embedding into this stream's collection
            milvus = u_data["milvus"]
            insert_data = [[global_var["counter"]], embeds, [str(meta.timestamp)]]
            milvus.insert(insert_data)
            
            # If necessary, insert the embedding into the global collection
            if sampling_fn():
                global_milvus = u_data["global_milvus"]
                insert_data = [embeds, [str(milvus.name)]]
                global_milvus.insert(insert_data)
                print("Inserted into global collection")
            insert_time = time.time()
            
            global_var["counter"] += 1  # Update frame counter
            
            # Log the latency
            if (DO_LATENCY_LOG):
                latency_log.write(str(global_var["counter"]) + "," +
                                str(((time_nanosec - (gst_buffer.pts - 37000000000)) / 1000000.)) + "," +
                                str((inference_time - start_time)*1000) + "," +
                                str((insert_time - inference_time)*1000) + "," +
                                str(gst_buffer.pts - 37000000000) + "," +
                                str(start_time) + "," +
                                str(inference_time) + "," +
                                str(insert_time) + "\n")
    return Gst.PadProbeReturn.OK


def init_collection(collection_name):
    if not utility.has_collection(collection_name): # Check if the collection already exists
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name="offset", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, "Video Stream")
        collection = Collection(collection_name, schema, consistency_level="Bounded")
        
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 64},
        }
        collection.create_index("embeddings", index)
    else:
        collection = Collection(collection_name)
    return collection


def init_global_collection():
    if not utility.has_collection("global"): # Check if the collection already exists
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name="collection", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, "Global Index")
        collection = Collection("global", schema, consistency_level="Bounded")
        
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 64},
        }
        collection.create_index("embeddings", index)
    else:   # Connect to existing collection
        collection = Collection("global")
    return collection


def main():
    parser = argparse.ArgumentParser(description='Pravega inferene job')
    parser.add_argument('--log_level', type=int, default=logging.INFO, help='10=DEBUG,20=INFO')
    parser.add_argument('--stream', default='urv6')
    parser.add_argument('--log_path', default=LOG_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    logging.info('args=%s' % str(args))
    
    log_path = args.log_path
    os.makedirs(log_path, exist_ok=True)  
    global latency_log 
    if (DO_LATENCY_LOG):
        latency_log = open(f"{log_path}/inference_log_{timestamp}.log", "a")
        latency_log.write("frame number,e2e latency(ms),model inference(ms),milvus transfer(ms),pts timestamp, initial timestamp, embedding timestamp, milvus timestamp\n")

    # Standard GStreamer initialization.
    Gst.init(None)
    logging.info(Gst.version_string())

    pipeline_description = (
        'pravegasrc name=src ! '
        'h264parse !'
        'avdec_h264 max_threads=1 ! '
        'videoconvert ! '
        'video/x-raw, format=RGB ! '
        'fakesink name=sink'
    )
    logging.info('Creating pipeline: ' +  pipeline_description)
    pipeline = Gst.parse_launch(pipeline_description)

    pravegasrc = pipeline.get_by_name('src')
    pravegasrc.set_property('controller', PRAVEGA_CONTROLLER)
    pravegasrc.set_property('stream', '%s/%s' % (PRAVEGA_SCOPE, args.stream))
    pravegasrc.set_property("start-mode", 'earliest')
    pravegasrc.set_property("end-mode", 'unbounded')
    pravegasrc.set_property("allow-create-scope", True)
    
    # Initialize the model
    model, device = get_model()
    
    # Connect to Milvus and initialize collections
    connections.connect(MILVUS_NAMESPACE, host=MILVUS_HOST, port=MILVUS_PORT)
    milvus_collection = init_collection(args.stream)
    milvus_global_collection = init_global_collection()

    sink = pipeline.get_by_name("sink")
    if sink:
        add_probe(pipeline, "sink", set_event_message_meta_probe, pad_name='sink', model=model, milvus=milvus_collection, global_milvus=milvus_global_collection, device=device)

    # Create an event loop and feed GStreamer bus messages to it.
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    # Start play back and listen to events.
    logging.info('Starting pipeline')
    pipeline_start = time.time()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        logging.error(traceback.format_exc())
        # Cleanup GStreamer elements.
        pipeline.set_state(Gst.State.NULL)
        raise
    
    pipeline_finish = time.time()
    
    # Logs and cleanup
    if (DO_LATENCY_LOG):
        latency_log.close()
    
    pipeline_duration = pipeline_finish - pipeline_start
    total_data = global_var["counter"]*global_var["size"][0]*global_var["size"][1]*global_var["size"][2]/1024/1024
    config = {
        "stream": args.stream,
        "log_path": log_path,
        "frame_res": global_var["size"],
        "frame_size_mb": global_var["size"][0]*global_var["size"][1]*global_var["size"][2]/1024/1024,
        "total_frames": global_var["counter"],
        "pipeline_duration_s": pipeline_duration,
        "total_data_mb": total_data,
        "throughput_mbps": total_data/pipeline_duration,
    }
    with open(f"{log_path}/inference_log_config_{timestamp}.json", "w") as f:
        json.dump(config, f)
        
    milvus_collection.flush()
    milvus_global_collection.flush()

    pipeline.set_state(Gst.State.NULL)
    logging.info('END')


if __name__ == '__main__':
    main()
