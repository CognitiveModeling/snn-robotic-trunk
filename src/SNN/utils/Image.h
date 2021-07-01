#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <inttypes.h>
#include <string>
#include <vector>
#include <assert.h>

namespace SNN {
    
    class Image {

        private:

            /* the witdh of this */
            int32_t width;

            /* the heigh of this */
            int32_t height;

            /* the pixel data of this */
            uint8_t  *pixel;

            /* the number of bytes per pixel */
            int8_t  bytesPerPixel;

            /* the total number of pixel */
            int64_t length;

            /* wether the image library was allready initialized */
            static bool initialized;

        public:

            /* constructor */
            Image(std::string);

            /* creat an image with the given width an height and byptes per pixel */
            Image(int32_t width, int32_t height, uint8_t bytesPerPixel = 1, int initialValue = 0);

            /* stack several 1D geryscale images */
            Image(std::vector<Image *>);

            /* opens all images from the given directory */
            static std::vector<Image *> open(std::string dir);

            /* destructor */
            ~Image();

            /* saves the image */
            void save(std::string);

            /* converts this to greyscale */
            void toGray();

            /* returns the width of this */
            int32_t getWidth() const { return width; }

            /* returns the height of this */
            int32_t getHeight() const { return height; }

            /* returns the bytes per pixel of this */
            int8_t getBytesPerPixel() const { return bytesPerPixel; }

            /* returns the total number of pixels of this */
            int32_t size() const { return length; }

            /* returns a pointer to the pixel */
            uint8_t *ptr() { return pixel; }

            /* acces operator for the pixels of this */
            uint8_t &operator [] (size_t i) { return pixel[i]; }

            /* returns the pixel at the given loacation */
            uint8_t &getPixel(int32_t x, int32_t y) {
                assert(x < width && y < height);
                return pixel[y * width + x];
            }

            /* returns the red pixel at the given loacation */
            uint8_t &getPixelRed(int32_t x, int32_t y) {
                assert(x < width && y < height);
                return pixel[(y * width + x) * 3];
            }

            /* returns the green pixel at the given loacation */
            uint8_t &getPixelGreen(int32_t x, int32_t y) {
                assert(x < width && y < height);
                return pixel[(y * width + x) * 3 + 1];
            }

            /* returns the blue pixel at the given loacation */
            uint8_t &getPixelBlue(int32_t x, int32_t y) {
                assert(x < width && y < height);
                return pixel[(y * width + x) * 3 + 2];
            }

            /* upscal this image about the given factor in x and y */
            void upscal(int factor);


    };
}

#endif
