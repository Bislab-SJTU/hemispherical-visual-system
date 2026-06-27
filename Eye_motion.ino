#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// ================= Hardware center calibration area =================
// Fill in the calibrated values for the eyeball center position.
// These are the same values that were originally written in setup().
#define X_CENTER  320   // Physical center of the horizontal servo
#define Y_CENTER  420   // Physical center of the vertical servo

// Motion range.
// This is the one-side swing range. A larger value makes the eyeball rotate farther.
#define RANGE     200
// ====================================================================

void setup() {
  Serial.begin(115200);

  pwm.begin();
  pwm.setPWMFreq(60);

  // 1. Initial reset to the calibrated center position.
  // This keeps the eyeball at the absolute center after power-on.
  pwm.setPWM(0, 0, X_CENTER);
  pwm.setPWM(1, 0, Y_CENTER);

  // 2. Lock the eyelids using calibrated fixed values.
  pwm.setPWM(2, 0, 300);
  pwm.setPWM(3, 0, 450);
  pwm.setPWM(4, 0, 440);
  pwm.setPWM(5, 0, 300);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');

    int xIndex = data.indexOf('X');
    int yIndex = data.indexOf('Y');

    if (xIndex != -1 && yIndex != -1) {
      int xVal = data.substring(xIndex + 1, yIndex).toInt();
      int yVal = data.substring(yIndex + 1).toInt();

      // ================= Jump-free mapping algorithm =================
      // Principle: use X_CENTER and Y_CENTER as the center values,
      // then extend RANGE to both sides.
      // Python sends 0    -> CENTER - RANGE
      // Python sends 500  -> CENTER approximately
      // Python sends 1000 -> CENTER + RANGE

      int lex = map(xVal, 0, 1023, X_CENTER - RANGE, X_CENTER + RANGE);
      int ley = map(yVal, 0, 1023, Y_CENTER - RANGE, Y_CENTER + RANGE);

      // Safety constraint to prevent the servo PWM from exceeding its limits.
      lex = constrain(lex, 150, 550);
      ley = constrain(ley, 150, 550);

      pwm.setPWM(0, 0, lex);
      pwm.setPWM(1, 0, ley);
    }
  }
}