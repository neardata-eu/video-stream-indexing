import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import json


def generate_fragments(frames, fragment_offset=5, similarity=0.9):
    """
    Generate fragments from a list of frames
    :param frames: a list of frames with the format (frame_number, similarity)
    :param fragment_offset: a number of frames to consider before and after the frame
    :param similarity: a threshold to consider a frame as a key frame
    :return: a list of tuples with the start and end of the fragments
    """
    # Remove frames with similarity below the threshold
    frames = [frame for frame in frames if frame[1] >= similarity]
    
    # Generate intervals around the frames
    intervals = [(frame[0] - fragment_offset, frame[0] + fragment_offset) for frame in frames]

    # Check if the intervals overlap and merge them
    merged_intervals = []
    for start, end in sorted(intervals):
        if merged_intervals and start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
        else:
            merged_intervals.append((start, end))

    return merged_intervals


def count_frames(filepath):
    """Count the number of frames in a video file using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-print_format', 'json',
        filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    frame_count = int(info['streams'][0]['nb_read_frames'])
    return frame_count


def process_files(filenames, result_path):
    """Count the number of frames in a list of files"""
    results = []
    for filename in filenames:
        if filename is not None:
            frame_count = count_frames(f"{result_path}/{filename}")
            if frame_count is not None:
                results.append({"filename": filename, "frame_count": frame_count})
            else:
                raise Exception(f'Error counting frames in file {filename}')
    return results