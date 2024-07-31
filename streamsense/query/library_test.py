from video_search import inter_video_search, intra_video_search
from PIL import Image

if __name__ == '__main__':
    ## Read our query image
    img = Image.open('/project/benchmarks/experiment3/cat_frame_ref.png')
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.resize((940, 560))
    
    print("Testing inter_video_search")
    candidates = inter_video_search(image=img, global_accuracy=0.8)
    print("Candidates found:")
    print(candidates)
    
    print("Testing intra_video_search")
    intra_video_search(image=img, global_accuracy=0.8, accuracy=0.85)