#include "TimeSubject.h"
struct ListNode
{
TimeObserver item;
struct ListNode* next;
};
static struct ListNode *observers;
//static SystemTime currentTime;
/* Local helper functions for managing the linked-list (implementation omitted). */
static void appendToList(const TimeObserver* observer)
{
/* Append a copy of the observer to the linked-list. */
if(observers==0){
struct ListNode *new_node=malloc(sizeof(struct ListNode));

new_node->item=*observer;
new_node->next=0;
observers=new_node;
}
else{
struct ListNode * node=observers;
while(node->next !=0){
node=node->next;
}
struct ListNode * listnode=malloc(sizeof(struct ListNode)) ;
listnode->item=* observer;
listnode->next=0;
node->next=listnode;
}

}
static void removeFromList(const TimeObserver* observer)
{
/* Identify the observer in the linked-list and remove that node. */
}
/* Implementation of the TimeSubject interface. */
void attach(const TimeObserver* observer)
{
//assert(NULL != observer);
appendToList(observer);
}
void detach(const TimeObserver* observer)
{
//assert(NULL != observer);
removeFromList(observer);
}
 void msTick()
{
struct ListNode* node = observers;
/* Invoke a function encapsulating the knowledge about time representation. */
int currentTime=5;
/* Walk through the linked-list and notify every observer that another millisecond passed. */
while(0 != node) {
TimeObserver* observer = &node->item;
observer->notification(observer->instance, currentTime);
node = node->next;
}
}
