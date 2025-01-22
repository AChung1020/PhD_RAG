from pymilvus import MilvusClient

client = MilvusClient("handbooks.db")


def create_collection(collection_name):
    if client.has_collection(collection_name="LGS-csi-handbooks"):
        client.drop_collection(collection_name="LGS-csi-handbooks")
    client.create_collection(
        collection_name="LGS-csi-handbooks",
        dimension=4096,
    )
