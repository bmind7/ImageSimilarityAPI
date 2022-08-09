# Image Similarity API
 
Our API utilizes a deep neural network to compare features of two images and calculate similarity score.

**Use cases**
* Find duplicated files on HDD - Reduces storage usage.
* Group images in a gallery - Users will not be overwhelmed with repeating content.
* Detect duplicated product images - Filter images during product creation in CMS.
* Similar product list - You can create a suggestion list by comparing images from each product within your current database.
* Copyrighted image detection

**How it works**
Neural network calculates the feature vector for each image. It then compares two feature vectors to figure how far apart two images are in latent space - apologies for the slang. In reality, it is simpler than it sounds. What is not simple is to achieve ease of use and performance. We continuously dedicate time to finding a perfect balance of precision and performance while simultaneously supporting hundreds of simultaneous requests per second. 

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

_Note about high concurrency support._
As you can see, we allow multiple concurrent requests on more expensive plans. It will enable the utilization of API much faster, but consequences can lead to some slowdown at the beginning of a session. We automatically scale our infrastructure to handle it when many new requests are received. It can lead to 10-15 seconds of waiting time until new instances of API are spawned. Most of the time, it should not affect other users, but if the queue of requests is full and we are spawning new instances, this can also lead to waiting time for other users.

**Limitations**
* Supported image formats: `bpm, gif, jpeg, png, pbm, tiff, tga, webp`.
* Request size should be `10MB` or less.
* URL should lead to images that are `10MB` or less.
* Request timeout for the image URLs is `2 seconds`.

**Tips & Tricks**
* To speed up response from API, send an image embedded into a query as `base64` content. That way server will not spend time downloading images from the web.
* Small images give pretty good results. By using an image of 256x256 size, you can save on bandwidth and increase the speed of sending requests to the API's endpoint. The reduction in precision will be insignificant.

**Release history**
1.1.0 - 25% speed improvement
1.0.2 - Fixed image alpha channel bug
1.0.1 - Improved concurrency of API instance
1.0.0 - Initial release