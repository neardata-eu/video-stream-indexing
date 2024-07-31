import time
from datetime import datetime
import argparse
from kubernetes import client, config
from kubernetes.client.rest import ApiException

def main(args):
    # Load Kubernetes configuration
    config.load_kube_config()

    # Define the namespace
    namespace = "default"

    # Job parameters
    git_repo_url = args.git_repo_url
    directory_to_enter = '/repo/scripts/inference/'
    stream = args.stream
    milvus_host = args.milvus_host
    commands = [
        "-c",
        (
            f"git clone {git_repo_url} /repo && "
            f"cd {directory_to_enter} && "
            f"GST_PLUGIN_PATH=/gstreamer-pravega/target/debug:${{GST_PLUGIN_PATH}} "
            f"python3 inference.py --stream {stream} --milvus_host {milvus_host}"
        )
    ]
    
    # Generate a timestamp and create a unique pod name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    pod_name = f"inference-pod-{timestamp}"

    # Define GPU resource requirements
    #resources = client.V1ResourceRequirements(
    #    requests={"nvidia.com/gpu": "1"},  # Request 1 GPU
    #    limits={"nvidia.com/gpu": "1"}     # Limit 1 GPU
    #)

    # Define the pod specification
    pod_spec = client.V1Pod(
        metadata=client.V1ObjectMeta(name=pod_name),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="inference-container",
                    image="arnaugabriel/video-indexing:2.0",
                    command=["/bin/bash"],
                    args=commands,
                    working_dir="/project",
                    #resources=resources,
                )
            ],
            restart_policy="Never",
            host_network=True,
        )
    )

    # Create an instance of the API class
    api_instance = client.CoreV1Api()

    try:
        # Create the pod
        api_instance.create_namespaced_pod(namespace=namespace, body=pod_spec)
        print("Pod created. Waiting for it to complete...")

        # Wait for the pod to complete
        while True:
            pod = api_instance.read_namespaced_pod(name=pod_name, namespace=namespace)
            if pod.status.phase in ["Succeeded", "Failed"]:
                print(f"Pod completed with status: {pod.status.phase}")
                break
            time.sleep(5)

        # Delete the pod
        api_instance.delete_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            body=client.V1DeleteOptions()
        )
        print("Pod deleted.")

    except ApiException as e:
        print("Exception when calling CoreV1Api: %s\n" % e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create and manage a Kubernetes pod for video indexing.')
    parser.add_argument('--git_repo_url', type=str, default='https://github.com/ArnauGabrielAtienza/video-stream-indexing', help='Project Repository')
    parser.add_argument('--stream', type=str, default='sample', help='Stream name')
    parser.add_argument('--milvus_host', type=str, default='172.28.1.1', help='Milvus host address')

    args = parser.parse_args()
    main(args)
