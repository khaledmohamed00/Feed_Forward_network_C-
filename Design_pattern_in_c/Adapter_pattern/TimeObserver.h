#ifndef TIMEROBSERVER_H
#define TIMEROBSERVER_H
typedef void (*ChangeTimeNotification)(void* instance, int newTime);
typedef struct
{
void* instance;
ChangeTimeNotification notification;
} TimeObserver;

#endif
