#ifndef FIGHTER_H
#define FIGHTER_H

typedef struct Fighter * Fighter_ptr;
typedef void (*SetJumpStratege_type)(Jump);
typedef void (*SetKickStratege_type)(Kick);

typedef void (*fun)(void);
struct Fighter{
fun punch;
fun kick;
fun roll;
fun jump;
SetJumpStratege_type SetJumpStratege;
SetKickStratege_type SetKickStratege;
};
Fighter_ptr Create_Fighter(fun Kick,fun jump);
#endif
