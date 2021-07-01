/**
 * simple linux and windows thread library
 */
#ifndef __THREADS_H__
#define __THREADS_H__

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <atomic>
#include <inttypes.h>

namespace SNN {

    /* abstract thread class to override */
    class Thread {
        
        private:

            /* thread function to override by child class */
            virtual void threadFunction(void *args) = 0;

            /* the actual thread function */
#ifdef _WIN32
            static void startThread(void *args);
#else
            static void *startThread(void *args);
#endif

            /* the custom thread args */
            void *args;

            /* thread handle */
#ifdef _WIN32
            HANDLE thread;
#else
            pthread_t thread;
#endif

        public:
            
            /* constructor */
            Thread() : args(NULL) {
#ifdef _WIN32
                thread = (HANDLE) 0;
#else
                thread = (pthread_t) 0;
#endif
            }

            /* start thread execution */
            void run(void *args = NULL);

            /* wait for thread execution to finish */
            void join();
            
            /* return an unqie thread identifier */
            static uint64_t id();

            /* handles execution to an other thread */
            static void yield();

            /* destructor */
            virtual ~Thread() { }

    };

    /* mutex class */
    class Mutex {
        
        private:

            /* the mutex object of this */
#ifdef _WIN32
            std::atomic_flag mutex = ATOMIC_FLAG_INIT;
#else
            pthread_mutex_t mutex;
#endif

        public:
            
            /* create a new Mutex */
            Mutex();

            /* delete a Mutex */
            ~Mutex();

            /* lock the mutex */
            void lock();

            /* try locking the mutex */
            bool trylock();

            /* unlock the mutex */
            void unlock();
    };

    /* bussy wait mutex for threads where sceduling cost would be high) */
    class FastMutex {
        
        private:

            /* the mutex object of this */
            std::atomic_flag mutex = ATOMIC_FLAG_INIT;

        public:
            
            /* lock the mutex */
            void lock() { while (mutex.test_and_set()) Thread::yield(); }

            /* try locking the mutex */
            bool trylock() { return mutex.test_and_set(); }

            /* unlock the mutex */
            void unlock() { mutex.clear(); };
    };

} 
    
#endif
