import json
import base64
import io
import requests
import asyncio
import time
import azure.functions as func
from asyncio.log import logger
from PIL import Image
from .ImageSimilarityNetONNX import modelONNX

MAX_REQUEST_LENGTH = 10_485_760
IMG_FETCH_TIMEOUT_SEC = 2

model_lock = asyncio.Lock()


async def main(req: func.HttpRequest) -> func.HttpResponse:
    headers = {
        "Content-type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    try:
        if 'warmup' in req.params:
            logger.info("Warm up!")
            return func.HttpResponse("ok")

        req_body = validate_and_get_request_body(req)

        eventloop = asyncio.get_event_loop()

        tasks = [
            eventloop.run_in_executor(
                None, get_pil_image, req_body['image_a']),
            eventloop.run_in_executor(
                None, get_pil_image, req_body['image_b'])
        ]
        done_tasks, _ = await asyncio.wait(tasks)
        img_a, img_b = map(lambda t: t.result(), done_tasks)

        sim_score = await get_similarity_score(eventloop, img_a, img_b)

        results = {'similarity_score': sim_score}
        status_code = 200
    except Exception as e:
        logger.error(str(e))
        results = {'error': str(e)}
        status_code = 400

    return func.HttpResponse(json.dumps(results), headers=headers, status_code=status_code)


def validate_and_get_request_body(req: func.HttpRequest):
    if req.get_body() == None:
        raise ValueError(f"Request body is empty")

    if len(req.get_body()) > MAX_REQUEST_LENGTH:
        raise ValueError(
            f"Request body is more than {MAX_REQUEST_LENGTH} bytes")

    try:
        req_body = req.get_json()
    except:
        raise ValueError("Can't parse request JSON body")

    # Check JSON structure
    if req_body['image_a'] == None or req_body['image_b'] == None or \
            len(req_body['image_a']) == 0 or len(req_body['image_b']) == 0:
        raise "Wrong JSON structure"

    return req_body


def get_pil_image(request_image: str):
    if request_image.startswith("http"):
        return process_url_attachment(request_image)
    else:
        return process_b64_attachment(request_image)


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
        return Image.open(io.BytesIO(content))
    except Exception as e:
        logger.error(str(e))
        raise ValueError("Can't read downloaded image")


def process_b64_attachment(content: str) -> Image:
    try:
        base64_decoded = base64.b64decode(content.split(',')[1])
        img = Image.open(io.BytesIO(base64_decoded))
        return img
    except Exception as e:
        logger.error(str(e))
        raise ValueError("Can't parse base64 image data")


async def get_similarity_score(eventloop, img_a, img_b):
    async with model_lock:
        inference_start_time = time.time()
        try:
            sim_score = await eventloop.run_in_executor(None, modelONNX.calculate, img_a, img_b)
        except Exception as e:
            logger.error(str(e))
            raise ValueError("Can't compare images")
        inference_end_time = time.time()

    inference_delta = inference_end_time - inference_start_time

    logger.info(
        f"Image similarity: {str(sim_score)} (inference time - {inference_delta * 1000:.1f}) ms")

    return sim_score
