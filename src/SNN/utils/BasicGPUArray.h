#ifndef __BASIC_GPU_ARRAY_H__
#define __BASIC_GPU_ARRAY_H__
#include <unordered_map>
#include "utils.h"
/**
 * template for a array shared on the cpu and gpu
 * with some helper functions for easy copying
 */
namespace SNN {

    class BasicGPUArray {
        
        public:

            /* destructor */
            virtual ~BasicGPUArray() { }

            /* copies the value to the gpu */
            virtual void copyToDevice() = 0;

            /* copies the value to the gpu */
            virtual void copyToDevice(int, int) = 0;

            /* copies the value to the cpu */
            virtual void copyToHost() = 0;
            virtual void copyToHost(int) = 0;
            virtual void copyToHost(int, int) = 0;
            virtual void copyToHost(int, int, int) = 0;

            /* returns the length of this */
            virtual int size() = 0;

            /* returns the global cuda memory consumption of this */
            virtual int globalMemoryConsumption() = 0;

            /* returns the device pointer */
            virtual void *d_ptr() const = 0;

            /* checking the values from the host arrays against the values from the device */
            virtual void check() = 0;

    };

    class NullGPUArray: public BasicGPUArray {

        public:

            /* copies the value to the gpu */
            virtual void copyToDevice() { }
            virtual void copyToDevice(int, int) { }

            /* copies the value to the cpu */
            virtual void copyToHost() { }
            virtual void copyToHost(int) { }
            virtual void copyToHost(int, int) { }
            virtual void copyToHost(int, int, int) { }

            /* returns the length of this */
            virtual int size() { return 0; }

            /* returns the global cuda memory consumption of this */
            virtual int globalMemoryConsumption() { return 0; }

            /* returns the device pointer */
            virtual void *d_ptr() const { return NULL; }

            /* checking the values from the host arrays against the values from the device */
            virtual void check() { }

    };

}
        

#endif /* __BASIC_GPU_ARRAY_H___ */
