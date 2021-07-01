#ifndef __GPU_ARRAY__
#define __GPU_ARRAY__
#include <unordered_map>
#include "BasicGPUArray.h"
#include "utils.h"
#include <vector>
/**
 * template for a array shared on the cpu and gpu
 * with some helper functions for easy copying
 */
namespace SNN {

    template <typename T> class GPUArray: public BasicGPUArray {
        
        private:
            
            /* the hots value of this */
            std::vector<T *> hostArrays;

            /* the device value of this */
            T *device;

            /* the size of this */
            int length;

            /* indicates that we own the host pointer */
            bool ownHost;

            /* frees the memory of this */
            void freeMemory() {

                if (length > 0) {
                    if (cudaFree(device))
                        log_err("cudaFree failed", LOG_EE);

                    if (ownHost)
                        for (auto &host: hostArrays)
                            delete host;

                    device = NULL;
                    hostArrays.clear();
                    length = 0;
                }
            }

            /* delete copy constructor */
            GPUArray(const GPUArray &other) = delete;

            /* delete  assignment operator */
            GPUArray &operator = (const GPUArray &other) = delete;

        public:

            /* constructor */
            GPUArray(T host) : length(1), device(NULL), ownHost(true) {
                this->hostArrays.push_back(new T);
                this->hostArrays[0][0] = host;

                if (cudaMalloc((void **) &device, sizeof(T) * length))
                    log_err("cudaMalloc malloc failed", LOG_EE);
            }

            GPUArray(T *host, int length) : 
                device(NULL), length(length), ownHost(false) {
                
                if (length > 0) {
                    hostArrays.push_back(host);

                    if (cudaMalloc((void **) &device, sizeof(T) * length))
                        log_err("cudaMalloc malloc failed", LOG_EE);
                }
            }

            GPUArray(std::vector<T *> hostArrays, int length) : 
                hostArrays(hostArrays), device(NULL), length(length), ownHost(false) {

                if (length > 0) {
                    if (cudaMalloc((void **) &device, sizeof(T) * length * hostArrays.size()))
                        log_err("cudaMalloc malloc failed", LOG_EE);
                }
            }

            /* destructor */
            ~GPUArray() {
                this->freeMemory();
            }

            /* copies the value to the gpu */
            void copyToDevice() {
                
                for (unsigned i = 0; i < hostArrays.size(); i++) {
                    if (hostArrays[i] != NULL) {
                        cudaError_t error = cudaMemcpy(
                            device + i * length, 
                            hostArrays[i], 
                            sizeof(T) * length, 
                            cudaMemcpyHostToDevice
                        );
                        if (error)
                            log_err("cudaMemcpyHostToDevice failed: " + itoa(error), LOG_EE);
                    }
                }
            }

            /* copies the value to the cpu */
            void copyToHost() {

                for (unsigned i = 0; i < hostArrays.size(); i++) {
                    if (hostArrays[i] != NULL) {
                        cudaError_t error = cudaMemcpy(
                            hostArrays[i], 
                            device + i * length, 
                            sizeof(T) * length, 
                            cudaMemcpyDeviceToHost
                        );
                        if (error)
                            log_err("cudaMemcpyDeviceToHost failed: " + itoa(error), LOG_EE);
                    }
                }
            }

            /* copies the array with the given index to the cpu */
            void copyToHost(int i) {
                
                if ((unsigned) i < hostArrays.size()) {
                    cudaError_t error = cudaMemcpy(
                        hostArrays[i], 
                        device + i * length, 
                        sizeof(T) * length, 
                        cudaMemcpyDeviceToHost
                    );
                    if (error)
                        log_err("cudaMemcpyDeviceToHost failed: " + itoa(error), LOG_EE);
                }
            }

            /* copies the array with the given index to the cpu */
            void copyToHost(int i, int start, int size) {
                
                if ((unsigned) i < hostArrays.size()) {
                    cudaError_t error = cudaMemcpy(
                        hostArrays[i] + start, 
                        device + i * length + start, 
                        sizeof(T) * size, 
                        cudaMemcpyDeviceToHost
                    );
                    if (error)
                        log_err("cudaMemcpyDeviceToHost failed: " + itoa(error), LOG_EE);
                }
            }

            /* copies the array with the given index to the cpu */
            void copyToHost(int start, int size) {
                
                for (unsigned i = 0; i < hostArrays.size(); i++) {
                    if (hostArrays[i] != NULL) {
                        cudaError_t error = cudaMemcpy(
                            hostArrays[i] + start, 
                            device + i * length + start, 
                            sizeof(T) * size, 
                            cudaMemcpyDeviceToHost
                        );
                        if (error)
                            log_err("cudaMemcpyDeviceToHost failed: " + itoa(error), LOG_EE);
                    }
                }
            }

            /* copies the array with the given index to the cpu */
            void copyToDevice(int start, int size) {
                
                for (unsigned i = 0; i < hostArrays.size(); i++) {
                    if (hostArrays[i] != NULL) {
                        cudaError_t error = cudaMemcpy(
                            device + i * length + start, 
                            hostArrays[i] + start, 
                            sizeof(T) * size, 
                            cudaMemcpyHostToDevice
                        );
                        if (error)
                            log_err("cudaMemcpyHostToDevice failed: " + itoa(error), LOG_EE);
                    }
                }
            }

            /* arry accessoperator for the type of this */
            T &at(int i) { 
                if ((unsigned) i < hostArrays.size()) 
                    return hostArrays[i / length][i % length]; 

                log_err("index out of range in GPUArray", LOG_E);
                return hostArrays[0][0];
            }

            /* returns the device pointer */
            void *d_ptr() const { return (void *) device; }

            /* returns the length of this */
            int size() { 
                return this->length;
            }

            /* returns the global cuda memory consumption of this */
            int globalMemoryConsumption() {
                return sizeof(T) * this->length;
            }

            /* checking the values from the host arrays against the values from the device */
            void check() {
                T *tmp = (T *) malloc(sizeof(T) * length);

                for (unsigned i = 0; i < hostArrays.size(); i++) {
                    if (hostArrays[i] != NULL) {
                        cudaError_t error = cudaMemcpy(
                            tmp,
                            device + i * length, 
                            sizeof(T) * length, 
                            cudaMemcpyDeviceToHost
                        );
                        if (error)
                            log_err("cudaMemcpyDeviceToHost failed: " + itoa(error), LOG_EE);

                        for (int n = 0; n < length; n++) {

                            FloatType v1 = tmp[n], v2 = hostArrays[i][n];
                            while (fabs(v1) > 1 || fabs(v2) > 1) {
                                v1 /= 2;
                                v2 /= 2;
                            }
                            if (fabs(v1 - v2) > 1e-3) {
                                log_err(
                                    "GPUArray value " + itoa(n) + " differs: host: " + 
                                    ftoa(hostArrays[i][n], 20) + ", device: " + 
                                    ftoa(tmp[n], 20) + ", error: " + 
                                    ftoa(fabs(tmp[n] - hostArrays[i][n]), 20), LOG_W);
                            }
                        }
                    }
                }

                free(tmp);
            }

    };

}
        

#endif /* __GPU_ARRAY__ */
