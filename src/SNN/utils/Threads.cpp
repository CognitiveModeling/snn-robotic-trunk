#include "Threads.h"
using namespace SNN;

#include <algorithm>
#include <memory.h>

#ifdef _WIN32
#include <process.h>
#endif


#ifdef _WIN32
void Thread::startThread(void *args) {
#else
void *Thread::startThread(void *args) {
#endif
    
    Thread *self = (Thread *) args;
    self->threadFunction(self->args);

#ifdef _WIN32
    _endthread();
#else
    return NULL;
#endif
}

void Thread::run(void *args) {
    this->args = args;

#ifdef _WIN32
    thread = (HANDLE) _beginthread(startThread, 0, (void *) this);
#else
    pthread_create(&thread, NULL, startThread, (void *) this);
#endif
}

void Thread::join() {
#ifdef _WIN32
    WaitForSingleObject(thread, INFINITE);
#else
    pthread_join(thread, NULL);
#endif
}

uint64_t Thread::id() {
#ifdef _WIN32
    return (uint64_t) GetCurrentThreadId();
#else
    pthread_t ptid = pthread_self();
    uint64_t threadId = 0;
    memcpy(&threadId, &ptid, std::min(sizeof(threadId), sizeof(ptid)));
    return threadId;
#endif
}

/* handles execution to an other thread */
void Thread::yield() {
#ifdef _WIN32
    SwitchToThread();
#else
    pthread_yield();
#endif
}

/* create a new Mutex */
Mutex::Mutex() {
#ifndef _WIN32
    pthread_mutex_init(&mutex, NULL);
#endif
}

/* delete a Mutex */
Mutex::~Mutex() {
#ifndef _WIN32
    pthread_mutex_destroy(&mutex);
#endif
}

/* lock the mutex */
void Mutex::lock() {
#ifdef _WIN32
    while (mutex.test_and_set());
#else
    pthread_mutex_lock(&mutex);
#endif
}

/* try locking the mutex */
bool Mutex::trylock() {
#ifdef _WIN32
    return mutex.test_and_set();
#else
    return pthread_mutex_trylock(&mutex);
#endif
}

/* unlock the mutex */
void Mutex::unlock() {
#ifdef _WIN32
    mutex.clear();
#else
    pthread_mutex_unlock(&mutex);
#endif
}

