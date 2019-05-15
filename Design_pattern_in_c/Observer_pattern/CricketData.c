#include"Observer.h"
#include "CricketData.h"
#include "Subject.h"
#include"stdlib.h"
//typedef CricketData * CricketData_ptr;
struct arraylist{
struct Observer_type *Observer;
struct arraylist *next;
};

struct CricketData {
int runs;
int wickets;
float overs;
struct arraylist *head;
};

/*typedef struct
{
void* instance;
ChangeTimeNotification notification;
} TimeObserver;
*/

CricketData_ptr create_CricketData(int runs,int wickets, float overs){
CricketData_ptr me=malloc(sizeof(struct CricketData));
me->runs=runs;
me->wickets=wickets;
me->overs=overs;
me->head=NULL;
return me;
}

void registerObserver(CricketData_ptr me,Observer_ptr  o)
{struct arraylist * Node=malloc(sizeof( struct arraylist));

if(me->head==NULL)
{me->head=Node;
me->head->Observer=o;
me->head->next=NULL;
}
else{
struct arraylist * node=me->head;
while(node->next !=NULL){
node=node->next;
}
struct arraylist * new_node=malloc(sizeof( struct arraylist));
new_node->Observer=o;
new_node->next=NULL;
node->next=new_node;

}
}

void notifyObservers(CricketData_ptr me)
{
struct arraylist * node=me->head;

while(node !=NULL){
node->Observer->update(node->Observer->instance,me->runs,me->wickets,me->overs);
node=node->next;
}
}


static int getLatestRuns()
{
        // return 90 for simplicity
        return 90;
}

    // get latest wickets from stadium
static int getLatestWickets()
{
        // return 2 for simplicity
        return 2;
}

    // get latest overs from stadium
static float getLatestOvers()
{
        // return 90 for simplicity
        return (float)10.2;
}

void dataChanged(CricketData_ptr me)
{
        //get latest data
        me->runs = getLatestRuns();
        me->wickets = getLatestWickets();
        me->overs = getLatestOvers();

        notifyObservers( me);
}
