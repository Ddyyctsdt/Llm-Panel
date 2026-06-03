import os
import time
import json
import logging
import requests
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
NAMESPACE_ID = os.environ.get("KV_NAMESPACE_ID")

if not ACCOUNT_ID or not API_TOKEN or not NAMESPACE_ID:
    logger.error("Missing required env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, KV_NAMESPACE_ID")
    sys.exit(1)

KV_API_BASE = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/storage/kv/namespaces/{NAMESPACE_ID}/values"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def kv_get(key: str) -> Optional[str]:
    for attempt in range(3):
        try:
            resp = requests.get(f"{KV_API_BASE}/{key}", headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                return None
            else:
                logger.warning(f"kv_get attempt {attempt+1} failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"kv_get attempt {attempt+1} error: {e}")
        time.sleep(1)
    return None

def kv_put(key: str, value: str, expiration_ttl: Optional[int] = None) -> bool:
    url = f"{KV_API_BASE}/{key}"
    params = {"expiration_ttl": expiration_ttl} if expiration_ttl is not None else {}
    for attempt in range(3):
        try:
            resp = requests.put(url, headers=HEADERS, data=value, params=params, timeout=10)
            if resp.status_code == 200:
                return True
            else:
                logger.warning(f"kv_put attempt {attempt+1} failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"kv_put attempt {attempt+1} error: {e}")
        time.sleep(1)
    return False

def kv_delete(key: str) -> bool:
    for attempt in range(3):
        try:
            resp = requests.delete(f"{KV_API_BASE}/{key}", headers=HEADERS, timeout=10)
            if resp.status_code in (200, 404):
                return True
            else:
                logger.warning(f"kv_delete attempt {attempt+1} failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"kv_delete attempt {attempt+1} error: {e}")
        time.sleep(1)
    return False

def update_response(response_id: str, new_chunk: str, is_done: bool = False) -> None:
    key = f"response:{response_id}"
    current = kv_get(key) or ""
    current += new_chunk
    if is_done:
        current += "[DONE]"
    kv_put(key, current, expiration_ttl=600)

def load_model():
    from llama_cpp import Llama
    model_path = "./model.gguf"
    if not os.path.exists(model_path):
        logger.info("Downloading model from Hugging Face...")
        url = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q6_K_XL.gguf?download=true"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model downloaded.")
    logger.info("Loading model with chat_format=qwen...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=2,
        verbose=False,
        chat_format="qwen"
    )
    logger.info("Model loaded.")
    return llm

def process_request(req_id: str, user_msg: str, llm) -> None:
    logger.info(f"Processing request {req_id}: {user_msg[:50]}...")
    messages = [{"role": "user", "content": user_msg}]
    start_time = time.time()
    try:
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        for chunk in stream:
            elapsed = time.time() - start_time
            if elapsed > 300:
                logger.error(f"Timeout for {req_id} after {elapsed:.1f}s")
                update_response(req_id, "\n[خطا: زمان پاسخ بیش از 5 دقیقه شد]", is_done=False)
                break
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    update_response(req_id, content, is_done=False)
        # پایان پاسخ (حتی اگر timeout شده باشد، DONE اضافه می‌شود)
        update_response(req_id, "", is_done=True)
        logger.info(f"Finished request {req_id}")
    except Exception as e:
        logger.error(f"Error processing {req_id}: {e}")
        update_response(req_id, f"\n[خطا: {str(e)}]", is_done=True)

def main():
    logger.info("Starting bot.py with create_chat_completion and qwen chat format")
    try:
        llm = load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    while True:
        try:
            queue_raw = kv_get("queue:next")
            if queue_raw:
                try:
                    queue_item = json.loads(queue_raw)
                    if queue_item.get("status") == "pending":
                        req_id = queue_item.get("id")
                        user_msg = queue_item.get("message")
                        # حذف فوری صف برای جلوگیری از race condition
                        if kv_delete("queue:next"):
                            logger.info(f"Acquired request {req_id}, processing...")
                            process_request(req_id, user_msg, llm)
                        else:
                            logger.warning("Could not delete queue:next, skipping")
                    else:
                        logger.debug(f"Queue item status {queue_item.get('status')} invalid, deleting")
                        kv_delete("queue:next")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in queue:next, deleting")
                    kv_delete("queue:next")
            else:
                logger.debug("Queue empty, sleeping 3s")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        time.sleep(3)

if __name__ == "__main__":
    main()
