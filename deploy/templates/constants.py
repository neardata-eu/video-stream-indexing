# Milvus config
MILVUS_HOST = "{{ hostvars[groups['milvus'][0]].private_ip }}"
MILVUS_PORT = "19530"
MILVUS_NAMESPACE = "default"

# Pravega config
PRAVEGA_CONTROLLER = "tcp://{{ hostvars[groups['bookkeeper'][0]].private_ip }}:9090"
PRAVEGA_SCOPE = "examples"

# Inference Logs
DO_LATENCY_LOG = True

# Path for all logs
LOG_PATH = "/project/logs"
RESULT_PATH = "/project/video_fragments"

if __name__ == "__main__":
    """Print the environment variables for the shell scripts"""
    print(f"export MILVUS_HOST={MILVUS_HOST}")
    print(f"export MILVUS_PORT={MILVUS_PORT}")
    print(f"export MILVUS_NAMESPACE={MILVUS_NAMESPACE}")
    print(f"export PRAVEGA_CONTROLLER={PRAVEGA_CONTROLLER}")
    print(f"export PRAVEGA_SCOPE={PRAVEGA_SCOPE}")
    print(f"export LOG_PATH={LOG_PATH}")
    print(f"export RESULT_PATH={RESULT_PATH}")