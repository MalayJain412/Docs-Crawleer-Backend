"""
Async helpers for uploading/downloading FAISS indexes (and other files)
to Azure Blob Storage. Supports:
 - AZURE_STORAGE_CONNECTION_STRING
 - Managed Identity: AZURE_STORAGE_ACCOUNT_URL + DefaultAzureCredential
Follow project's import fallback pattern and async-first style.
"""

try:
    from config.settings import settings
    from utils.logger import default_logger
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import settings
    from utils.logger import default_logger

import os
import asyncio
from typing import Optional

# Async SDKs
from azure.storage.blob.aio import BlobServiceClient
from azure.identity.aio import DefaultAzureCredential

DEFAULT_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "faiss-indexes")


async def _get_blob_service_client() -> (BlobServiceClient, Optional[DefaultAzureCredential]):
    """
    Create an async BlobServiceClient.
    Returns (client, credential_or_None) so credential can be closed when done.
    """
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if conn_str:
        client = BlobServiceClient.from_connection_string(conn_str)
        return client, None

    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    if not account_url:
        raise RuntimeError(
            "Azure storage not configured. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL"
        )

    cred = DefaultAzureCredential()
    client = BlobServiceClient(account_url=account_url, credential=cred)
    return client, cred


async def upload_file(local_path: str, container_name: str = DEFAULT_CONTAINER, blob_name: Optional[str] = None):
    """
    Async upload a file to blob storage. Overwrites existing blob.
    """
    if blob_name is None:
        blob_name = os.path.basename(local_path)

    client, cred = await _get_blob_service_client()
    try:
        container_client = client.get_container_client(container_name)
        try:
            await container_client.create_container()
        except Exception:
            # container may already exist
            pass

        # open file in binary mode and upload
        with open(local_path, "rb") as data:
            await container_client.upload_blob(name=blob_name, data=data, overwrite=True)

        default_logger.info(f"Uploaded {local_path} -> container '{container_name}' blob '{blob_name}'")
    finally:
        await client.close()
        if cred:
            await cred.close()


async def download_file(local_path: str, container_name: str = DEFAULT_CONTAINER, blob_name: Optional[str] = None):
    """
    Async download a blob to local_path. Overwrites local file if exists.
    """
    if blob_name is None:
        blob_name = os.path.basename(local_path)

    client, cred = await _get_blob_service_client()
    try:
        container_client = client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)

        stream = await blob_client.download_blob()
        data = await stream.readall()

        # Ensure parent dir exists
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)

        default_logger.info(f"Downloaded blob '{blob_name}' -> {local_path}")
    finally:
        await client.close()
        if cred:
            await cred.close()


# Synchronous wrappers for code that expects blocking IO
def upload_file_sync(local_path: str, container_name: str = DEFAULT_CONTAINER, blob_name: Optional[str] = None):
    asyncio.run(upload_file(local_path, container_name=container_name, blob_name=blob_name))


def download_file_sync(local_path: str, container_name: str = DEFAULT_CONTAINER, blob_name: Optional[str] = None):
    asyncio.run(download_file(local_path, container_name=container_name, blob_name=blob_name))