#include <TimerThree.h>

bool level = 0;
void rts();

void setup() {
Timer3.initialize(1000); // Cada 1ms
Timer3.attachInterrupt(rts);
pinMode(LED_BUILTIN,OUTPUT);
}

void rts(){
  static unsigned long counter = 0;
  counter++;
    if(counter%500==0){
    level = level^1;
    digitalWrite(LED_BUILTIN,level);
  }
}

void loop() {
  }
