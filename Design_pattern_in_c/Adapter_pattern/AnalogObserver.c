#include "TimeObserver.h"
#include "AnalogObserver.h"
struct AnalogStopWatch
{
 int analog_time;
/* Other attributes of the watch, e.g. digital display. */
};
static void updateDisplay(AnalogStopWatchPtr AnalogWatch,int newTime){

AnalogWatch->analog_time=newTime;
printf("Analog clock time %d\n",newTime);

}
/* Implementation of the function required by the TimeObserver interface. */
static void changedTime(void* instance, int newTime)
{
AnalogStopWatchPtr AnalogWatch = instance;
updateDisplay(AnalogWatch, newTime);
}


AnalogStopWatchPtr createAnalogWatch(void)
{
AnalogStopWatchPtr watch = malloc(sizeof(struct AnalogStopWatch));
if(0 != watch){
/* Successfully created -> attach to the subject. */
TimeObserver observer = {0};
observer.instance = watch;
observer.notification = changedTime;
attach(&observer);
}
return watch;
}
