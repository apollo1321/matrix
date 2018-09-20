#include "threadpool.h"
#include <pthread.h>
#include <iostream>

ThreadPool::ThreadPool(int N) : threadCount(N), queueLock(PTHREAD_MUTEX_INITIALIZER), queueCond(PTHREAD_COND_INITIALIZER)
{
    threads = new pthread_t[N];
    for(int i = 0; i < N; ++i){
        pthread_create(&threads[i], NULL, threadExecute, (void*)this);
    }

}

bool ThreadPool::loadJob(JobInterface* &aJob)
{
    pthread_mutex_lock(&queueLock);
    while(jobQueue.empty()){
        pthread_cond_wait(&queueCond, &queueLock);
    }
    aJob = jobQueue.front();
    jobQueue.pop();
    pthread_mutex_unlock(&queueLock);
    return true;
}

void* ThreadPool::threadExecute(void *arg)
{
    ThreadPool* self = (ThreadPool*)arg;
    JobInterface* aJob = nullptr;
    while( self->loadJob(aJob) ){
        aJob->working();
        delete aJob;
        aJob = nullptr;
    }
    return nullptr;
}

void ThreadPool::assignJob(JobInterface *aJob)
{
    pthread_mutex_lock(&queueLock);
    jobQueue.push(aJob);
    pthread_mutex_unlock(&queueLock);
    pthread_cond_signal(&queueCond);
}

ThreadPool::~ThreadPool(){
    pthread_mutex_lock(&queueLock);
    while(!jobQueue.empty()){
        delete jobQueue.front();
        jobQueue.pop();
    }
    for(int i = 0; i < threadCount; ++i){
        pthread_cancel(threads[i]);
    }

    pthread_mutex_destroy(&queueLock);
    pthread_cond_destroy(&queueCond);

    delete[] threads;

}

