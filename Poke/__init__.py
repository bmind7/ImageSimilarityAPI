import datetime
import logging
import requests
import azure.functions as func


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    # if mytimer.past_due:
    #     logging.info('The timer is past due!')

    requests.get(
        "https://move37-image-analysis-api.azurewebsites.net/api/imagesimilarityindex")
    logging.info('Poke timer trigger function ran at %s', utc_timestamp)
