#include <TimerThree.h>

bool level = 0;
bool encender = false;
bool sendFlag = false;

void rts();

void setup() {
Serial.begin(115200);
Timer3.initialize(1000); // Cada 1ms
Timer3.attachInterrupt(rts);
pinMode(LED_BUILTIN,OUTPUT);
}

void rts(){
  static unsigned long counter = 0;
  counter++;

  if(encender){
    if(counter%500==0){
    level = level^1;
    digitalWrite(LED_BUILTIN,level);
  }
  }
else{
  digitalWrite(LED_BUILTIN,0);
}
}

void loop() {
  
  if(Serial.available()>0){
    sendFlag = true;
  }
  if(sendFlag){
    char data = Serial.read();
    if(data == 'e'){
      encender = true;
      sendFlag = false;
    }
    else if (data == 'a'){
      encender = false;
      sendFlag = false;
    }
  }
}
