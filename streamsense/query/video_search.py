import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import (
    connections,
    MilvusClient
)

from policies.constants import (MILVUS_HOST, MILVUS_PORT, MILVUS_NAMESPACE)
from policies.components import get_model, inference
from index_utils import search, search_global

from PIL import Image
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

def milvus_connection(milvus_host=MILVUS_HOST, milvus_namespace=MILVUS_NAMESPACE, milvus_port=MILVUS_PORT):
    connections.connect(milvus_namespace, host=milvus_host, port=milvus_port)
    client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
    return client

def get_embedding(img):
    model, device = get_model()
    embeds = inference(model, np.array(img), device)  # Get embeddings
    return embeds

def inter_video_search(image=None, embedding=None,
                       global_k=5, global_accuracy=0.0, global_f=20,
                       client=None, milvus_host=MILVUS_HOST, milvus_namespace=MILVUS_NAMESPACE, milvus_port=MILVUS_PORT):
    
    ## Connect to Milvus
    if client==None:
        client = milvus_connection(milvus_host, milvus_namespace, milvus_port)

    if embedding==None:
        ## Initialize embedding model
        model, device = get_model()
        embedding = inference(model, np.array(image), device)  # Get embeddings

    ## Search global index
    candidates, _ = search_global("global", embedding.detach().numpy(), ["collection"], int(global_k), float(global_accuracy), global_f)
    return candidates


def intra_video_search(image=None, embedding=None, result_path='/project/results',
                       global_accuracy=0.0,
                       parallelism_candidates=5, local_k=100, fragment_offset=10, accuracy=0.9, parallelism_exports=5,
                       client=None, milvus_host=MILVUS_HOST, milvus_namespace=MILVUS_NAMESPACE, milvus_port=MILVUS_PORT):
    
    ## Connect to milvus
    if client==None:
        client = milvus_connection(milvus_host, milvus_namespace, milvus_port)
        
    if embedding==None:
        ## Initialize embedding model
        model, device = get_model()
        embedding = inference(model, np.array(image), device)  # Get embeddings
    
    ## Search global index
    candidates = inter_video_search(image=image, embedding=embedding, client=client, milvus_host=milvus_host, milvus_namespace=milvus_namespace, milvus_port=milvus_port, global_accuracy=global_accuracy)
    
    ## Perform queries
    fragments = []
    output_fields=["offset", "pk"]
    with ThreadPoolExecutor(max_workers=int(parallelism_candidates)) as executor:
        search_partial = partial(
            search, 
            milvus_client=client, 
            embedding=embedding.detach().numpy(), 
            fields=output_fields, 
            local_k=int(local_k),
            fragment_offset=int(fragment_offset), 
            accuracy=float(accuracy), 
            result_path=result_path,
            parallelism=int(parallelism_exports)
        )
        futures = {executor.submit(search_partial, collection_name=collection_name): collection_name for collection_name in candidates}

        for future in as_completed(futures):
            fragment, _, _ = future.result()
            fragments.extend(fragment)