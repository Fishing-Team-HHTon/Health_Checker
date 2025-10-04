#include <Arduino.h>

const uint8_t ECG_PIN    = A0;
const uint8_t LO_NEG_PIN = 2;
const uint8_t LO_POS_PIN = 3;
const uint8_t PPG_PIN    = A1;
const uint8_t RESP_PIN   = A2;

enum Mode { MODE_ECG, MODE_PPG, MODE_RESP, MODE_EMG };
volatile Mode modeSel = MODE_ECG;

const uint32_t BAUD = 115200;

const uint32_t PERIOD_US_ECG  = 10000; // 100 Гц
const uint32_t PERIOD_US_PPG  = 10000; // 100 Гц
const uint32_t PERIOD_US_RESP = 10000; // 100 Гц
const uint32_t PERIOD_US_EMG  =  2000; // 500 Гц

inline int median3(int a, int b, int c) {
  if (a > b) { int t=a; a=b; b=t; }
  if (b > c) { int t=b; b=c; c=t; }
  if (a > b) { int t=a; a=b; b=t; }
  return b;
}

const float ALPHA_RESP = 0.12f;   
const float ALPHA_PPG  = 0.20f;   
const float ALPHA_EMG_ENV = 0.25f;
const float ALPHA_BASE_SLOW = 0.01f; 

float emaResp = 0.0f, emaPPG = 0.0f, emgEnv = 0.0f, baseEMG = 512.0f, basePPG = 512.0f;
int   respP1=0, respP2=0, ppgP1=0, ppgP2=0, emgP1=0, emgP2=0;

void resetFilters() {
  emaResp = emaPPG = emgEnv = 0.0f;
  baseEMG = basePPG = 512.0f;
  respP1 = respP2 = ppgP1 = ppgP2 = emgP1 = emgP2 = 0;
}

void setMode(Mode m) {
  modeSel = m;
  resetFilters();
}

void maybeSwitchMode() {
  while (Serial.available()) {
    char c = tolower(Serial.read());
    if (c == 'e') setMode(MODE_ECG);
    if (c == 'p') setMode(MODE_PPG);
    if (c == 'r') setMode(MODE_RESP);
    if (c == 'm') setMode(MODE_EMG);
  }
}

int readECG() {
  bool leadOff = (digitalRead(LO_NEG_PIN) == HIGH) || (digitalRead(LO_POS_PIN) == HIGH);
  if (leadOff) return 0;
  return analogRead(ECG_PIN);
}

int readPPG() {
  int x0 = analogRead(PPG_PIN);
  int x  = median3(x0, ppgP1, ppgP2); ppgP2 = ppgP1; ppgP1 = x0;
  basePPG = (1.0f - ALPHA_BASE_SLOW)*basePPG + ALPHA_BASE_SLOW*(float)x;
  float xdc = (float)x - basePPG;                 
  emaPPG = (1.0f - ALPHA_PPG)*emaPPG + ALPHA_PPG*xdc;
  int out = (int)(emaPPG + 512.0f);               
  if (out < 0) out = 0; if (out > 1023) out = 1023;
  return out;
}

int readResp() {
  int x0 = analogRead(RESP_PIN);
  int x  = median3(x0, respP1, respP2); respP2 = respP1; respP1 = x0;
  emaResp = (1.0f - ALPHA_RESP)*emaResp + ALPHA_RESP*(float)x;
  int out = (int)emaResp;
  if (out < 0) out = 0; if (out > 1023) out = 1023;
  return out;
}

int readEMG() {
  int x0 = analogRead(ECG_PIN);
  int x  = median3(x0, emgP1, emgP2); emgP2 = emgP1; emgP1 = x0;
  baseEMG = (1.0f - ALPHA_BASE_SLOW)*baseEMG + ALPHA_BASE_SLOW*(float)x;
  float xdc = (float)x - baseEMG;
  float rect = fabs(xdc);
  emgEnv = (1.0f - ALPHA_EMG_ENV)*emgEnv + ALPHA_EMG_ENV*rect;
  int out = (int)emgEnv;
  if (out < 0) out = 0; if (out > 1023) out = 1023;
  return out;
}

void setup() {
  Serial.begin(BAUD);
  pinMode(ECG_PIN,    INPUT);
  pinMode(LO_NEG_PIN, INPUT);
  pinMode(LO_POS_PIN, INPUT);
  pinMode(PPG_PIN,    INPUT);
  pinMode(RESP_PIN,   INPUT);
  resetFilters();
}

void loop() {
  maybeSwitchMode();

  static uint32_t next_us = 0;
  const uint32_t now = micros();

  uint32_t period = PERIOD_US_ECG;
  if (modeSel == MODE_PPG)  period = PERIOD_US_PPG;
  if (modeSel == MODE_RESP) period = PERIOD_US_RESP;
  if (modeSel == MODE_EMG)  period = PERIOD_US_EMG;

  if ((int32_t)(now - next_us) >= 0) {
    next_us = now + period;

    int v = 0;
    switch (modeSel) {
      case MODE_ECG:  v = readECG();  break;
      case MODE_PPG:  v = readPPG();  break;
      case MODE_RESP: v = readResp(); break;
      case MODE_EMG:  v = readEMG();  break;
    }

    Serial.println(v);
  }
}
