#include <stdio.h>
#include <stdlib.h>
#include "jump.h"
#include "Kick.h"
#include "Fighter.h"
#include "KEN.h"
#include "RYU.h"
#include "JumpStratege.h"
#include "KickStratege.h"
int main()
{ KEN_ptr ken=Create_KEN_Fighter(TornadoKick,LongJump);
  ken->display();
  ken->super->kick();
  ken->super->jump();

  RYU_ptr ryu=Create_RYU_Fighter(LightningKick,ShortJump);
  ryu->display();
  ryu->super->kick();
  ryu->super->jump();

    //printf("Hello world!\n");
    return 0;
}
