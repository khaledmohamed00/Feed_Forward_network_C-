#include "TimeObserver.h"
#include "DigitalObserver.h"
struct DigitalStopWatch
{
 int digital_time;
/* Other attributes of the watch, e.g. digital display. */
};
/* Implementation of the function required by the TimeObserver interface. */
static void updateDisplay(DigitalStopWatchPtr digitalWatch,int newTime){

digitalWatch->digital_time=newTime;
printf("Digital clock time %d\n",newTime);

}

static void changedTime(void* instance, int newTime)
{
DigitalStopWatchPtr digitalWatch = instance;
updateDisplay(digitalWatch, newTime);
}



DigitalStopWatchPtr createDigitalWatch(void)
{
DigitalStopWatchPtr watch = malloc(sizeof(struct DigitalStopWatch));
if(0 != watch){
/* Successfully created -> attach to the subject. */
TimeObserver observer = {0};
observer.instance = watch;
observer.notification = changedTime;
attach(&observer);
}
return watch;
}
