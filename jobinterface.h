#ifndef JOBINTERFACE_H
#define JOBINTERFACE_H

class JobInterface
{
public:
    JobInterface(){ ; }
    virtual void working() = 0;
    virtual ~JobInterface(){ ; }
};


#endif // JOBINTERFACE_H
