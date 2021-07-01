#include "Image.h"
#include <IL/il.h>
#include <pthread.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <dirent.h>
#include "utils.h"
#include <string.h>

using namespace SNN;

/* wether the image library was allready initialized */
bool Image::initialized = false;

/* synconization */
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/* constructor */
Image::Image(std::string fname) {

    if (!initialized) {
        ilInit();
        initialized = true;
    }

    pthread_mutex_lock(&mutex);
    if (ilLoadImage(fname.c_str()) == IL_FALSE) {
        pthread_mutex_unlock(&mutex);
        assert(false);
    }
    
    this->width         = ilGetInteger(IL_IMAGE_WIDTH);
    this->height        = ilGetInteger(IL_IMAGE_HEIGHT);
    this->bytesPerPixel = ilGetInteger(IL_IMAGE_BITS_PER_PIXEL) / 8;
    this->length        = this->width * this->height * this->bytesPerPixel;
    this->pixel         = (uint8_t *) malloc(this->length);
    
    memcpy(this->pixel, ilGetData(), this->length);
 
    if (ilGetError() != IL_NO_ERROR) {
        free(this->pixel);
        free(this);
        pthread_mutex_unlock(&mutex);
        assert(false);
    }
 
    pthread_mutex_unlock(&mutex);

    /* mirror horizontaly */
    uint8_t *tmp = (uint8_t *) malloc(this->bytesPerPixel * width);
    for (int32_t i = 0; i < height / 2; i++) {
        memcpy(tmp, this->pixel + i * this->bytesPerPixel * width, this->bytesPerPixel * width);
        memcpy(
            this->pixel + i * this->bytesPerPixel * width, 
            this->pixel + (height - i - 1) * this->bytesPerPixel * width, 
            this->bytesPerPixel * width
        );
        memcpy(
            this->pixel + (height - i - 1) * this->bytesPerPixel * width, 
            tmp,
            this->bytesPerPixel * width
        );
    }
    free(tmp);
}

/* creat an image with the given width an height and byptes per pixel */
Image::Image(int32_t width, int32_t height, uint8_t bytesPerPixel, int initialValue) {

    if (!initialized) {
        ilInit();
        initialized = true;
    }

    this->width         = width;
    this->height        = height;
    this->bytesPerPixel = bytesPerPixel;
    this->length        = this->width * this->height * this->bytesPerPixel;
    this->pixel         = (uint8_t *) malloc(this->length);
    memset(this->pixel, initialValue, this->length);
}

/* stack several 1D greyscale images */
Image::Image(std::vector<Image *> imgs): Image(imgs[0]->width, imgs.size(), 1) {
    
    if (!initialized) {
        ilInit();
        initialized = true;
    }

    for (unsigned y = 0; y < imgs.size(); y++) 
        for (int x = 0; x < width; x++) 
            pixel[y * width + x] = imgs[y]->pixel[x];
}

/* destructor */
Image::~Image() {
    free(this->pixel);
}

/* saves the image */
void Image::save(std::string fname) {
    pthread_mutex_lock(&mutex);
    unlink(fname.c_str());
    ilTexImage(
        this->width,
        this->height,
        1,
        this->bytesPerPixel,
        (this->bytesPerPixel == 1) ? IL_LUMINANCE : ((this->bytesPerPixel == 3) ? IL_RGB : IL_RGBA),
        IL_UNSIGNED_BYTE,
        this->pixel
    );
    ilSaveImage(fname.c_str());
    pthread_mutex_unlock(&mutex);
}

/* converts this to greyscale */
void Image::toGray() {
    assert(this->bytesPerPixel == 3);
  
    int64_t i, j;
    for (i = 0, j = 0; i < this->length; i += 3, j++) {
        this->pixel[j] = this->pixel[i]     * 0.299 +
                         this->pixel[i + 1] * 0.587 +
                         this->pixel[i + 2] * 0.114;
    }
 
    this->length /= 3;
    this->pixel = (uint8_t *) realloc(this->pixel, this->length);
    this->bytesPerPixel = 1;
}

/* opens all images from the given directory */
std::vector<Image *> Image::open(std::string dir) {

    if (!initialized) {
        ilInit();
        initialized = true;
    }

    if (dir.back() != '/')
        dir += '/';

    std::vector<Image *> images;

    DIR * dirp;
    struct dirent *entry;

    dirp = opendir(dir.c_str());
    if (dirp == NULL) {
        perror("Couldn't open the directory");
        exit(EXIT_FAILURE);
    }

    while ((entry = readdir(dirp)) != NULL)
        if (entry->d_type == DT_REG) {
            std::string file = dir + std::string(entry->d_name);

        if (endsWith(file, ".jpg") || endsWith(file, ".png") ||
            endsWith(file, ".bmp") || endsWith(file, ".gif"))
            images.push_back(new Image(file));
    }
          
    closedir(dirp);
    return images;
}

/* upscal this image about the given factor in x and y */
void Image::upscal(int factor) {
    if (this->bytesPerPixel == 1) {
        unsigned oldWidth  = this->width;
        this->width *= factor;
        this->height *= factor;
        this->length = this->width * this->height;
        uint8_t *tmpPixel = (uint8_t *) malloc(this->length);

        for (int x = 0;  x < this->width; x += factor) {
            for (int y = 0;  y < this->height; y += factor) {
                for (int i = 0; i < factor; i++) {
                    memset(
                        tmpPixel + (y + i) * this->width + x,
                        this->pixel[(y / factor) * oldWidth + x / factor],
                        factor
                    );
                }
            }
        }
        
        free(this->pixel);
        this->pixel = tmpPixel;
    } else {

        this->width *= factor;
        this->height *= factor;
        this->length = this->width * this->height * this->bytesPerPixel;
        uint8_t *tmpPixel = (uint8_t *) malloc(this->length);

        for (int x = 0;  x < this->width; x += factor) {
            for (int y = 0;  y < this->height; y += factor) {
                for (int xx = 0; xx < factor; xx++) {
                    for (int yy = 0; yy < factor; yy++) {
                        memcpy(
                            tmpPixel + ((y + yy) * this->width + x + xx) * this->bytesPerPixel, 
                            this->pixel + (y * this->width / (factor * factor) + x / factor) * this->bytesPerPixel,
                            this->bytesPerPixel
                        );
                    }
                }
            }
        }

        free(this->pixel);
        this->pixel = tmpPixel;
    }
}
