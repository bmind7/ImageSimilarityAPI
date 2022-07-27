# Image Similarity API
 
Our API utilizes a deep neural network to compare features of two images and calculate similarity score.

**Use cases**
* Find duplicated files on HDD - Reduces storage usage.
* Group images in a gallery - Users will not be overwhelmed with repeating content.
* Detect duplicated product images - Filter images during product creation in CMS.
* Similar product list - You can create a suggestion list by comparing images from each product within your current database.
* Copyrighted image detection

**How it works**
Neural network calculates the feature vector for each image. It then compares two feature vectors to figure how far apart two images are in latent space - apologies for the slang. In reality, it is simpler than it sounds. What is not simple is to achieve ease of use and performance. We spent many hours finding a perfect balance of precision and performance. Also, we continuously dedicate time to researching how to make inference faster for our products.

*Please contact us if you need an image's feature vector. We plan to release such API in nearest future.*

**How to use**
Send a request with a JSON body of the following structure:
```
{
    "image_a": "url or base64 encoded image",
    "image_b": "url or base64 encoded image",
}
```

*Please note that the URL image should allow hotlinking; otherwise, we will not be able to reach it.*

As a response, you will get JSON with `similarity_score` in the range from `0.0` to `1.0`. Where `0.0` is the least similar possible.

**Limitations**
* Supported image formats: `bpm, gif, jpeg, png, pbm, tiff, tga, webp`.
* Request size should be `10MB` or less.
* URL should lead to images that are `10MB` or less.
* Public plans support 1 request per second. This limitation is in place because of the high computational intensity of the current task. Please contact us if you need a highly parallel solution with 10-20-50 and more simultaneous requests. We will be able to set up a dedicated endpoint for you.

**Tips & Tricks**
* To speed up response from API, send an image embedded into a query as `base64` content. That way server will not spend time downloading images from the web.
* Small images give pretty good results. By using an image of 256x256 size, you can save on bandwidth and increase the speed of sending requests to the API's endpoint. The reduction in precision will be insignificant.

**Release history**
1.0.2 - Fixed image alpha channel bug
1.0.1 - Improved concurrency of API instance
1.0.0 - Initial release