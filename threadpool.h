#ifndef THREADPOOL_H
#define THREADPOOL_H

#include "jobinterface.h"
#include <queue>
#include <pthread.h>

class ThreadPool
{
    std::queue<JobInterface*> jobQueue;
    const int threadCount;
    pthread_t* threads;
    pthread_mutex_t queueLock;
    pthread_cond_t queueCond;

    static void* threadExecute(void *arg);

    bool loadJob(JobInterface*& aJob);

public:
    ThreadPool(int N);
    void assignJob(JobInterface *aJob);
    int getThreadCount() const{
        return threadCount;
    }
    ~ThreadPool();
};

#endif // THREADPOOL_H
