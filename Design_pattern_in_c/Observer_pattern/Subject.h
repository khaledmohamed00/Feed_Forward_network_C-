#ifndef SUBJECT_H
#define SUBJECT_H

void registerObserver(CricketData_ptr me,Observer_ptr o);
void unregisterObserver(void *o);
void notifyObservers(CricketData_ptr me);

#endif
