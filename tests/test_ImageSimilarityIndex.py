import azure.functions as func
import pytest
from ImageSimilarityIndex import main


@pytest.mark.asyncio
async def test_get_request_empty_body():
    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='GET',
        body=None,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body() == b'{"error": "Request body is empty"}'


@pytest.mark.asyncio
async def test_get_wrong_warmpup_request():
    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='GET',
        body=None,
        url='/api/imagesimilarityindex?warmuppppp=1')

    resp = await main(req)

    assert resp.get_body() == b'{"error": "Request body is empty"}'


@pytest.mark.asyncio
async def test_get_correct_warmpup_request():
    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='GET',
        body=None,
        params={'warmup': 1},
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body() == b'ok'


@pytest.mark.asyncio
async def test_post_request_empty_body():
    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=None,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body() == b'{"error": "Request body is empty"}'


@pytest.mark.asyncio
async def test_post_request_body_too_big():
    with open("./tests/20mb.jpg", "rb") as f:
        im_bytes = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=im_bytes,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(b'{"error": "Request body is more than')


@pytest.mark.asyncio
async def test_post_request_empty_body():
    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=b'',
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(
        b'{"error": "Can\'t parse request JSON body')


@pytest.mark.asyncio
async def test_post_url_incorrect():
    with open("./tests/url_incorrect.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(b'{"error": "Can\'t download image"')


@pytest.mark.asyncio
async def test_post_base64_incorrect():
    with open("./tests/base64_incorrect.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(
        b'{"error": "Can\'t parse base64 image data"')


@pytest.mark.asyncio
async def test_post_url_image_too_big():
    with open("./tests/url_image_too_big.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(
        b'{"error": "Image file is bigger than allowed limit')


@pytest.mark.asyncio
async def test_post_url_and_base64_correct():
    with open("./tests/url_and_base64_correct.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(b'{"similarity_score":')


@pytest.mark.asyncio
async def test_post_url_and_url_correct():
    with open("./tests/url_and_url_correct.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(b'{"similarity_score":')


@pytest.mark.asyncio
async def test_post_base64_and_base64_correct():
    with open("./tests/base64_and_base64_correct.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(b'{"similarity_score":')


@pytest.mark.asyncio
async def test_post_base64_and_url_correct():
    with open("./tests/base64_and_url_correct.json", "rb") as f:
        json = f.read()

    # Construct a mock HTTP request.
    req = func.HttpRequest(
        method='POST',
        body=json,
        url='/api/imagesimilarityindex')

    resp = await main(req)

    assert resp.get_body().startswith(b'{"similarity_score":')
