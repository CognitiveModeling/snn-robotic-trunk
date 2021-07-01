#ifndef __BASIC_SERIALIZER__
#define __BASIC_SERIALIZER__
#include <vector>
#include <memory>
#include "utils.h"

namespace SNN {

    class BasicSerializer {
        
        public:

            /* destructor */
            virtual ~BasicSerializer() { }

            /* should serialize this */
            virtual void serialize(bool gpuMode = false) = 0;

            /* should deserialize this */
            virtual void deserialize(bool gpuMode = false) = 0;

            /* should check for correct values */
            virtual void check(bool gpuMode = false) { (void) gpuMode; }

            /* should save neccesarry values for model restoring to the given file */
            virtual void save(int) { }

            /* should load neccesarry values for model restoring from the given file */
            virtual void load(int) { }

            /* saves this as an image */
            virtual void saveAsImage(
                std::string, 
                unsigned, 
                unsigned, 
                unsigned 
            ) {
                log_err("saveAsImage not implemented in child class!!", LOG_E);
            }
    };

}

#endif
