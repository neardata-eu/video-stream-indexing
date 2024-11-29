from PIL import Image
from query.video_search import inter_video_search, intra_video_search
from multiprocessing import Process
from kubernetes import client, config
from time import sleep
from policies.components import get_model

banner = """
███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗███████╗███████╗███╗   ██╗███████╗███████╗
██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║███████╗█████╗  ██╔██╗ ██║███████╗█████╗  
╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  
███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████║███████╗██║ ╚████║███████║███████╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
"""


help_text = """ Commands may be of the following types:
\tFOR STREAM s USE EMBEDDINGS MODEL m INDEX SAMPLING i
\tGET k STREAMS FROM f FRAMES x % LIKE [IMAGE] USE EMBEDDINGS MODEL m
\tFROM k STREAMS GET f FRAGMENTS WITH FRAMES x % LIKE [IMAGE] FRAGMENT_LENGTH t USE EMBEDDINGS MODEL m"""

error_text = "Invalid command. Type 'help' for help."

indexing_policies = {}



def submit_indexing_job(s, m, i):
    key = (s, m, i)
    indexing_policies[key] = "ACTIVE"
    
    # Submit the k8s job
    config.load_kube_config()
    pod_name = "indexer-{}-{}-{}".format(s, m, i)
    namespace = "default"
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name},
        "spec": {
            "containers": [
                {
                    "name": "task-container",
                    "image": "arnaugabriel/video-indexing:2.0",  
                    "command": ["sh", "-c", "GST_PLUGIN_PATH=/gstreamer-pravega/target/debug:${GST_PLUGIN_PATH} python3 inference.py --stream {s} --model {m} --num_frames {i}"],
                }
            ],
            "restartPolicy": "Never",
        },
    }
    v1 = client.CoreV1Api()
    v1.create_namespaced_pod(namespace=namespace, body=pod_manifest)
    
    # Wait for the pod to finish
    while True:
        pod_status = v1.read_namespaced_pod_status(name=pod_name, namespace=namespace)
        if pod_status.status.phase in ["Succeeded", "Failed"]:
            print(f"Pod {pod_name} finished with status {pod_status.status.phase}")
            break
        sleep(5)
    
    # Update the indexing policy
    indexing_policies[key] = "COMPLETED"

def index_stream(s, m, i):
    print("Indexing stream {} using embeddings model {} with index sampling {}".format(s, m, i))
    
    # Submit the indexing job
    Process(target=submit_indexing_job, args=(s, m, i)).start()
    
def load_image(image):
    img = Image.open(image)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.resize((940, 560))
    return img

def load_model(m: str):
    return get_model(m)

def get_streams(k: str, f: str, x: str, image: str, m: str):
    print("Getting {} streams from {} frames with {}% like {} using embeddings model {}".format(k, f, x, image, m))
    
    img = load_image(image)    
    global_k = int(k)
    global_f = int(f)
    global_x = float(x)
    model,device = load_model(m)
    
    streams = inter_video_search(image=img, global_k=global_k, global_accuracy=global_x, global_f=global_f, model=model, device=device)
    
    print("Obtained streams: ")
    for stream in streams:
        print(f"\t- {stream}")

def get_fragments(k: str, f: str, x: str, image: str, t: str, m: str):
    print("Getting {} streams from {} frames with {}% like {} with fragment length {} using embeddings model {}".format(k, f, x, image, t, m))
    
    img = load_image(image)
    k = int(k)
    f = int(f)
    x = float(x)
    t = int(t)
    model, device = load_model(m)
    
    inter_video_search(image=img, local_k=k, fragment_offset=f, accuracy=x, fragment_length=t, model=model, device=device)
    print("Fragments obtained and saved in the result path defined in the constants file.")
    
    

def execute_command(command):
    command_parts = command.split()
    if command_parts[0] == "FOR":
        index_stream(command_parts[2], command_parts[6], command_parts[8])
    elif command_parts[0] == "GET":
        get_streams(command_parts[1], command_parts[4], command_parts[6], command_parts[9], command_parts[13])
    elif command_parts[0] == "FROM":
        get_fragments(command_parts[1], command_parts[4], command_parts[8], command_parts[11], command_parts[13], command_parts[17])    
    elif command_parts[0] == "help":
        print(help_text)
    else:
        print(error_text)
        
def get_command():
    print("> ", end="")
    command = input()
    return command


def main():
    print(banner)
    print("---------------------------------")
    print("Welcome to StreamSense CLI")
    print("Type 'exit' to exit the CLI")
    print("---------------------------------")
    
    command = get_command()
    
    while command != "exit":
        try:
            execute_command(command)
            print()
        except:
            print("An error occurred while executing the command. Type 'help' for help.")
            
        command = get_command()



if __name__ == '__main__':
    main()