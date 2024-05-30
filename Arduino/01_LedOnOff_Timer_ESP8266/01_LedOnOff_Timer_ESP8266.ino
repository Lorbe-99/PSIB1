#include <Ticker.h>

Ticker timer;
bool level = 0;

void rts() {
  static unsigned long counter = 0;
  counter++;
  if (counter % 500 == 0) {
    level = !level;
    digitalWrite(LED_BUILTIN, level);
  }
}

void setup() {
  Serial.begin(115200);
  timer.attach_ms(1, rts); // Llama a la función rts() cada 1ms
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // No es necesario colocar código en el loop(), ya que rts() se ejecutará en las interrupciones del temporizador
}
