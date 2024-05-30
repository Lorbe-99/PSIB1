
bool sendFlag = false;

void rts();

void setup() {
Serial.begin(115200);
pinMode(LED_BUILTIN,OUTPUT);
}

void loop() {
  
  if(Serial.available()>0){
    sendFlag = true;
  }
  if(sendFlag){
    char data = Serial.read();
    if(data == 'e'){
      digitalWrite(LED_BUILTIN,1);
      sendFlag = false;
    }
    else if (data == 'a'){
      digitalWrite(LED_BUILTIN,0);
      sendFlag = false;
    }
  }
}
