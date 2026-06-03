import os
import time
import json
import logging
import requests
import sys
from typing import Optional, Generator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
NAMESPACE_ID = os.environ.get("KV_NAMESPACE_ID")

if not ACCOUNT_ID or not API_TOKEN or not NAMESPACE_ID:
    logger.error("Missing required environment variables: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, KV_NAMESPACE_ID")
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
                logger.warning(f"kv_get attempt {attempt+1} failed: {resp.status_code} {resp.text}")
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
                logger.warning(f"kv_put attempt {attempt+1} failed: {resp.status_code} {resp.text}")
        except Exception as e:
            logger.warning(f"kv_put attempt {attempt+1} error: {e}")
        time.sleep(1)
    return False

def kv_delete(key: str) -> bool:
    for attempt in range(3):
        try:
            resp = requests.delete(f"{KV_API_BASE}/{key}", headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                return True
            elif resp.status_code == 404:
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
        url = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Model downloaded.")
    logger.info("Loading model...")
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False)
    logger.info("Model loaded.")
    return llm

def call_model_stream(prompt: str, llm) -> Generator[str, None, None]:
    try:
        # بررسی پشتیبانی از stream
        if hasattr(llm, 'create_completion') and 'stream' in llm.create_completion.__code__.co_varnames:
            logger.info("Using real streaming mode")
            stream = llm.create_completion(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stream=True,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            for chunk in stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        yield delta['content']
                        time.sleep(0.2)
        else:
            logger.info("Using simulated streaming (full generation then split)")
            response = llm.create_completion(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            full_text = response['choices'][0]['text']
            chunk_size = 10
            for i in range(0, len(full_text), chunk_size):
                yield full_text[i:i+chunk_size]
                time.sleep(0.5)
    except Exception as e:
        logger.error(f"Model generation error: {e}")
        yield f"\n[خطا در تولید: {e}]"

def process_request(req_id: str, user_msg: str, llm) -> None:
    logger.info(f"Processing request {req_id}: {user_msg[:50]}...")
    prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    start_time = time.time()
    try:
        full_response = ""
        for chunk in call_model_stream(prompt, llm):
            elapsed = time.time() - start_time
            if elapsed > 300:  # 5 minutes
                logger.error(f"Timeout for request {req_id} after {elapsed:.1f} seconds")
                update_response(req_id, f"\n[خطا: زمان پاسخ بیش از 5 دقیقه شد]", is_done=False)
                break
            full_response += chunk
            update_response(req_id, chunk, is_done=False)
        update_response(req_id, "", is_done=True)
        logger.info(f"Finished request {req_id} (length: {len(full_response)} chars)")
    except Exception as e:
        logger.error(f"Unexpected error processing {req_id}: {e}")
        update_response(req_id, f"\n[خطا: {str(e)}]", is_done=True)

def main():
    logger.info("Starting bot.py (fixed race condition + timeout)")
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
                        deleted = kv_delete("queue:next")
                        if not deleted:
                            logger.warning("Could not delete queue:next, maybe already taken by another action? Skipping.")
                        else:
                            logger.info(f"Acquired and deleted queue item {req_id}, processing...")
                            process_request(req_id, user_msg, llm)
                    else:
                        logger.debug(f"Queue item status is {queue_item.get('status')}, ignoring")
                        kv_delete("queue:next")  # پاک کردن کلید نامعتبر
                except json.JSONDecodeError:
                    logger.error("Invalid JSON in queue:next, deleting it")
                    kv_delete("queue:next")
            else:
                logger.debug("Queue empty, sleeping 3s")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        time.sleep(3)

if __name__ == "__main__":
    main()
