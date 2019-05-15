#include "jump.h"
#include "Kick.h"
#include "Fighter.h"

void (*jumpstratege)(void);
void (*kickstratege)(void);

void SetJumpStratege(fun jump){
jumpstratege=jump;
}

void SetKickStratege(fun kick){
kickstratege=kick;
}

void punch(void)
{
        printf("Default Punch\n");
}
void kick(void)
{
        // delegate to kick behavior
        (*kickstratege)();
}
void jump(void)
{


        // delegate to jump behavior
(*jumpstratege)();

}
void roll(void)
{
        printf("Default Roll\n");
}


Fighter_ptr Create_Fighter(fun Kick_fun,fun jump_fun){

Fighter_ptr me=(Fighter_ptr)malloc(sizeof (struct Fighter));


me->punch=punch;
me->kick=kick;
me->roll=roll;
me->jump=jump;
me->SetJumpStratege=SetJumpStratege;
me->SetKickStratege =SetKickStratege;
me->SetJumpStratege(jump_fun);
me->SetKickStratege(Kick_fun);
return me;
}

