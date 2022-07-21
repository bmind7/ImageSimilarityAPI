import json
import base64
import io
import requests
import azure.functions as func
from asyncio.log import logger
from PIL import Image
from .ImageSimilarityNet import model

MAX_REQUEST_LENGTH = 10_485_760


def main(req: func.HttpRequest) -> func.HttpResponse:
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

        if(req_body['image_a']['type'] == "url"):
            img = process_url_attachment(req_body['image_a']['content'])
        else:
            img = process_b64_attachment(req_body['image_a']['content'])

        if(req_body['image_b']['type'] == "url"):
            img2 = process_url_attachment(req_body['image_b']['content'])
        else:
            img2 = process_b64_attachment(req_body['image_b']['content'])

        sim_index = model.calculate(img, img2)

        logger.info("Image similarity: " + str(sim_index))
        results = {'sim_index': sim_index}
        status_code = 200
    except Exception as e:
        logger.error(str(e))
        results = {'error': str(e)}
        status_code = 400

    return func.HttpResponse(json.dumps(results), headers=headers, status_code=status_code)


def process_url_attachment(url: str) -> Image:
    try:
        # TODO: timeout
        # TODO: size check
        response = requests.get(url, stream=True)
        return Image.open(response.raw)
    except:
        raise ValueError("Can't download image from provider URL")


def process_b64_attachment(content: str) -> Image:
    try:
        base64_decoded = base64.b64decode(content.split(',')[1])
        img = Image.open(io.BytesIO(base64_decoded))
        return img
    except:
        raise ValueError("Can't parse base64 image data")
