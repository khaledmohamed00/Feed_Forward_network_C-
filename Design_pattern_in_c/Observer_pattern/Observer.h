#ifndef OBSERVER_H
typedef void (*fun)(void *,int,int,float);
typedef struct Observer_type * Observer_ptr;

 struct Observer_type
{
void* instance;
fun update;
} ;
//void update(void *o,int runs, int wickets,
//                      float overs);
#endif
