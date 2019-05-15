#include <stdlib.h>
#include<stdio.h>
#include "Observer.h"
#include "AverageScoreDisplay.h"
//typedef void (*fun)(void *,int,int,float);
 struct AverageScoreDisplay{
float runRate;
int predictedScore;
//fun update;
};

AverageScoreDisplay_ptr create_AverageScoreDisplay(void)
{
AverageScoreDisplay_ptr me=malloc(sizeof(struct AverageScoreDisplay));
//me->fun=&update
return me;

}
static void update(void *o,int runs, int wickets,float overs)
{
AverageScoreDisplay_ptr me=(AverageScoreDisplay_ptr)o;
me->runRate =(float)runs/overs;
me->predictedScore = (int)(me->runRate * 50);

printf("runRate=%f predictedScore=%d\n",me->runRate,me->predictedScore);

}
 Observer_ptr AverageScoreDisplay_ptr_to_Observer_ptr(AverageScoreDisplay_ptr me){

  Observer_ptr o=(Observer_ptr)malloc(sizeof(struct Observer_type));
  o->instance=me;
  o->update=&update;
  return o;
 }


