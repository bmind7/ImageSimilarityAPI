import json
import base64
import io
import requests
import asyncio
import time
import azure.functions as func
from asyncio.log import logger
from PIL import Image
from .ImageSimilarityNet import model

MAX_REQUEST_LENGTH = 10_485_760
IMG_FETCH_TIMEOUT_SEC = 5

model_lock = asyncio.Lock()


async def main(req: func.HttpRequest) -> func.HttpResponse:
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    try:
        if req.get_body() == None:
            raise ValueError(f"Request body is empty")

        if len(req.get_body()) > MAX_REQUEST_LENGTH:
            raise ValueError(
                f"Request body is more than {MAX_REQUEST_LENGTH} bytes")

        try:
            req_body = req.get_json()
            # Check JSON structure
            if req_body['image_a'] == None or req_body['image_b'] == None or \
               len(req_body['image_a']) == 0 or len(req_body['image_b']) == 0:
                raise "Wrong JSON structure"
        except:
            raise ValueError("Can't parse request JSON body")

        eventloop = asyncio.get_event_loop()

        if req_body['image_a'].startswith("http"):
            img = await eventloop.run_in_executor(None, process_url_attachment, req_body['image_a'])
        else:
            img = process_b64_attachment(req_body['image_a'])

        if req_body['image_b'].startswith("http"):
            img2 = await eventloop.run_in_executor(None, process_url_attachment, req_body['image_b'])
        else:
            img2 = process_b64_attachment(req_body['image_b'])

        async with model_lock:
            inference_start_time = time.time()
            try:
                sim_score = await eventloop.run_in_executor(None, model.calculate, img, img2)
            except Exception as e:
                logger.error(str(e))
                raise ValueError("Can't compare images")
            inference_end_time = time.time()

        inference_delta = inference_end_time - inference_start_time

        logger.info(
            f"Image similarity: {str(sim_score)} (inference time - {inference_delta * 1000:.1f}) ms")
        results = {'similarity_score': sim_score}
        status_code = 200
    except Exception as e:
        logger.error(str(e))
        results = {'error': str(e)}
        status_code = 400

    return func.HttpResponse(json.dumps(results), headers=headers, status_code=status_code)


def process_url_attachment(url: str) -> Image:
    try:
        response = requests.get(
            url, stream=True, timeout=IMG_FETCH_TIMEOUT_SEC)
    except Exception as e:
        logger.error(f"URL: {url[:200]}")
        logger.error(str(e))
        raise ValueError("Can't download image")

    if response.status_code != 200:
        logger.error(f"URL: {url[:200]}")
        logger.error(f"Response code: {response.status_code}")
        raise ValueError("Can't download image")

    content = response.raw.read(MAX_REQUEST_LENGTH + 1)
    if len(content) > MAX_REQUEST_LENGTH:
        logger.error(f"URL: {url[:200]}")
        raise ValueError(
            "Image file is bigger than allowed limit ({MAX_REQUEST_LENGTH} bytes)")

    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        logger.error(str(e))
        raise ValueError("Can't read downloaded image")


def process_b64_attachment(content: str) -> Image:
    try:
        base64_decoded = base64.b64decode(content.split(',')[1])
        img = Image.open(io.BytesIO(base64_decoded)).convert("RGB")
        return img
    except Exception as e:
        logger.error(str(e))
        raise ValueError("Can't parse base64 image data")
