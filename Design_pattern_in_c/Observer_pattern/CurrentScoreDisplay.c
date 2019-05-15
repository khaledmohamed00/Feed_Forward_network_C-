#include "Observer.h"
#include "CurrentScoreDisplay.h"

//typedef void (*fun)(void *,int,int,float);
 struct CurrentScoreDisplay {
 int runs, wickets;
 float overs;
//fun update;
};

CurrentScoreDisplay_ptr create_CurrentScoreDisplay(void)
{
CurrentScoreDisplay_ptr me=malloc(sizeof(struct CurrentScoreDisplay));
//me->fun=&update
return me;

}
static void update(void *o,int runs, int wickets,float overs)
{
CurrentScoreDisplay_ptr me=(CurrentScoreDisplay_ptr)o;
me->runs =runs;
me->wickets =wickets;
me->overs =overs;


printf("runs=%d wickets=%d overs=%f\n",me->runs,me->wickets,me->overs);

}

Observer_ptr CurrentScoreDisplay_ptr_to_Observer_ptr(CurrentScoreDisplay_ptr me){

  Observer_ptr o=(Observer_ptr)malloc(sizeof(struct Observer_type));
  o->instance=me;
  o->update=&update;
  return o;
 }


