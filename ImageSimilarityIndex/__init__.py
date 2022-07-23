import json
import base64
import io
import requests
import asyncio
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
        if(len(req.get_body()) > MAX_REQUEST_LENGTH):
            raise ValueError(
                f"Request body is more than {MAX_REQUEST_LENGTH} bytes")

        try:
            req_body = req.get_json()
            # Check JSON structure
            if req_body['image_a']['type'] == None or \
               req_body['image_b']['type'] == None or \
               req_body['image_a']['content'] == None or \
               req_body['image_b']['content'] == None:
                raise "Wrong JSON structure"
        except:
            raise ValueError("Can't parse request JSON body")

        eventloop = asyncio.get_event_loop()

        if(req_body['image_a']['type'] == "url"):
            img = await eventloop.run_in_executor(None, process_url_attachment, req_body['image_a']['content'])
        else:
            img = process_b64_attachment(req_body['image_a']['content'])

        if(req_body['image_b']['type'] == "url"):
            img2 = await eventloop.run_in_executor(None, process_url_attachment, req_body['image_b']['content'])
        else:
            img2 = process_b64_attachment(req_body['image_b']['content'])

        async with model_lock:
            sim_index = await eventloop.run_in_executor(None, model.calculate, img, img2)

        logger.info("Image similarity: " + str(sim_index))
        results = {'sim_index': sim_index}
        status_code = 200
    except Exception as e:
        logger.error(str(e))
        results = {'error': str(e)}
        status_code = 400

    return func.HttpResponse(json.dumps(results), headers=headers, status_code=status_code)


def process_url_attachment(url: str) -> Image:
    response = requests.get(url, stream=True, timeout=IMG_FETCH_TIMEOUT_SEC)

    content = response.raw.read(MAX_REQUEST_LENGTH + 1)
    if len(content) > MAX_REQUEST_LENGTH:
        raise ValueError(
            "Image file is bigger than allowed limit ({MAX_REQUEST_LENGTH} bytes)")

    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except:
        raise ValueError("Can't read downloaded image")


def process_b64_attachment(content: str) -> Image:
    try:
        base64_decoded = base64.b64decode(content.split(',')[1])
        img = Image.open(io.BytesIO(base64_decoded)).convert("RGB")
        return img
    except:
        raise ValueError("Can't parse base64 image data")
