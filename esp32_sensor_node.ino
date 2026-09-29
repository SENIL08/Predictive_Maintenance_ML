#include <WiFi.h>
#include <ThingSpeak.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <math.h>

// =====================================================
// PIN DEFINITIONS
// =====================================================

#define DHT_PIN 4
#define DHT_TYPE DHT11

#define ACS_PIN 35

#define PROXIMITY_PIN 27

#define SDA_PIN 21
#define SCL_PIN 22

// =====================================================
// WIFI + THINGSPEAK
// =====================================================

const char* ssid = "SAGAR";
const char* password = "sagar123";

unsigned long myChannelNumber = 3461983;
const char* myWriteAPIKey = "STIB2OOZEYZWPNEJ";

WiFiClient client;

// =====================================================
// SENSOR OBJECTS
// =====================================================

DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_MPU6050 mpu;

// =====================================================
// MEASUREMENT SETTINGS
// =====================================================

// 10 seconds
const unsigned long measurementInterval = 10000;

// =====================================================
// RPM SETTINGS
// =====================================================

// One proximity pulse = one revolution
const int targetsPerRevolution = 1;

volatile unsigned long pulseCount = 0;

// =====================================================
// SENSOR VARIABLES
// =====================================================

float temperature = 0.0;

float vibration = 0.0;
float vibration_mm_s = 0.0;

float current = 0.0;
float rpm = 0.0;

// =====================================================
// PROXIMITY INTERRUPT
// =====================================================

void IRAM_ATTR countPulse()
{
  pulseCount++;
}

// =====================================================
// ACS712-5A CURRENT RMS
// =====================================================

float readCurrent()
{
  // ACS712-5A sensitivity = 185 mV/A
  const float ACS_SENSITIVITY = 0.185;

  const int samples = 500;

  double sumVoltage = 0.0;
  double sumVoltageSquare = 0.0;

  for (int i = 0; i < samples; i++)
  {
    int adcValue = analogRead(ACS_PIN);

    float voltage =
      (adcValue * 3.3) / 4095.0;

    sumVoltage += voltage;
    sumVoltageSquare += voltage * voltage;

    delayMicroseconds(100);
  }

  float meanVoltage =
    sumVoltage / samples;

  float variance =
    (sumVoltageSquare / samples) -
    (meanVoltage * meanVoltage);

  if (variance < 0)
  {
    variance = 0;
  }

  float acVoltageRMS =
    sqrt(variance);

  float currentRMS =
    acVoltageRMS / ACS_SENSITIVITY;

  return currentRMS;
}

// =====================================================
// SETUP
// =====================================================

void setup()
{
  Serial.begin(115200);

  delay(1000);

  Serial.println();
  Serial.println();
  Serial.println("========================================");
  Serial.println(" ESP32 MOTOR PREDICTIVE MAINTENANCE");
  Serial.println("========================================");
  Serial.println("ESP32 Started!");
  Serial.println("Serial communication OK");
  Serial.println();

  // ===================================================
  // DHT11
  // ===================================================

  Serial.println("Initializing DHT11...");

  dht.begin();

  Serial.println("DHT11 initialized.");

  // ===================================================
  // MPU6050
  // ===================================================

  Serial.println("Initializing MPU6050...");

  Wire.begin(SDA_PIN, SCL_PIN);

  if (!mpu.begin())
  {
    Serial.println("ERROR: MPU6050 NOT DETECTED!");

    while (1)
    {
      Serial.println("Waiting for MPU6050...");
      delay(2000);
    }
  }

  Serial.println("MPU6050 detected!");

  mpu.setAccelerometerRange(
    MPU6050_RANGE_8_G
  );

  mpu.setFilterBandwidth(
    MPU6050_BAND_94_HZ
  );

  // ===================================================
  // PROXIMITY SENSOR
  // ===================================================

  Serial.println("Initializing proximity sensor...");

  pinMode(
    PROXIMITY_PIN,
    INPUT_PULLUP
  );

  attachInterrupt(
    digitalPinToInterrupt(PROXIMITY_PIN),
    countPulse,
    FALLING
  );

  Serial.println("Proximity sensor initialized.");

  // ===================================================
  // ADC
  // ===================================================

  analogReadResolution(12);

  Serial.println("ADC initialized.");

  // ===================================================
  // WIFI
  // ===================================================

  WiFi.mode(WIFI_STA);

  WiFi.begin(
    ssid,
    password
  );

  Serial.print("Connecting to WiFi");

  unsigned long wifiStart =
    millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - wifiStart < 20000
  )
  {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.println("WiFi Connected!");

    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  }
  else
  {
    Serial.println("WiFi connection FAILED!");
    Serial.println("Continuing without WiFi...");
  }

  // ===================================================
  // THINGSPEAK
  // ===================================================

  ThingSpeak.begin(client);

  Serial.println();
  Serial.println("========================================");
  Serial.println("SYSTEM READY");
  Serial.println("========================================");
  Serial.println("Measurement Interval : 10 Seconds");
  Serial.println("Cloud Update          : 10 Seconds");
  Serial.println("RPM Measurement       : 10 Seconds");
  Serial.println("RPM Formula           : Pulses x 6");
  Serial.println("ACS712                : 5A Version");
  Serial.println("Current Sensitivity   : 185 mV/A");
  Serial.println("Vibration Cloud       : mm/s RMS");
  Serial.println("Vibration Frequency   : 55 Hz");
  Serial.println();
  Serial.println("ThingSpeak Fields:");
  Serial.println("Field 1 = Temperature");
  Serial.println("Field 2 = Vibration");
  Serial.println("Field 3 = Current");
  Serial.println("Field 4 = RPM");
  Serial.println("========================================");
}

// =====================================================
// LOOP
// =====================================================

void loop()
{
  Serial.println();
  Serial.println();
  Serial.println("========================================");
  Serial.println("STARTING 10 SECOND MEASUREMENT");
  Serial.println("========================================");

  // ===================================================
  // RESET RPM COUNTER
  // ===================================================

  noInterrupts();

  pulseCount = 0;

  interrupts();

  // ===================================================
  // START 10 SECOND TIMER
  // ===================================================

  unsigned long measurementStart =
    millis();

  unsigned long lastStatusPrint =
    measurementStart;

  // ===================================================
  // TEMPERATURE
  // ===================================================

  float tempReading =
    dht.readTemperature();

  if (!isnan(tempReading))
  {
    temperature = tempReading;
  }

  Serial.print("Temperature: ");
  Serial.print(temperature, 2);
  Serial.println(" °C");

  // ===================================================
  // VIBRATION RMS VARIABLES
  // =====================================================

  double sumX = 0.0;
  double sumY = 0.0;
  double sumZ = 0.0;

  double sumXSquare = 0.0;
  double sumYSquare = 0.0;
  double sumZSquare = 0.0;

  unsigned long vibrationSamples = 0;

  // ===================================================
  // 10 SECOND SENSOR MEASUREMENT
  // ===================================================

  Serial.println();
  Serial.println("Collecting vibration + RPM data...");

  while (
    millis() - measurementStart <
    measurementInterval
  )
  {
    // =================================================
    // READ MPU6050
    // =================================================

    sensors_event_t a;
    sensors_event_t g;
    sensors_event_t temp;

    mpu.getEvent(
      &a,
      &g,
      &temp
    );

    // =================================================
    // X AXIS
    // =================================================

    sumX += a.acceleration.x;

    sumXSquare +=
      (double)a.acceleration.x *
      (double)a.acceleration.x;

    // =================================================
    // Y AXIS
    // =================================================

    sumY += a.acceleration.y;

    sumYSquare +=
      (double)a.acceleration.y *
      (double)a.acceleration.y;

    // =================================================
    // Z AXIS
    // =================================================

    sumZ += a.acceleration.z;

    sumZSquare +=
      (double)a.acceleration.z *
      (double)a.acceleration.z;

    vibrationSamples++;

    // =================================================
    // STATUS EVERY 1 SECOND
    // =================================================

    if (
      millis() - lastStatusPrint >= 1000
    )
    {
      unsigned long elapsed =
        millis() - measurementStart;

      noInterrupts();

      unsigned long livePulses =
        pulseCount;

      interrupts();

      Serial.print("Time: ");
      Serial.print(elapsed / 1000);
      Serial.print("/10 sec | Pulses: ");
      Serial.println(livePulses);

      lastStatusPrint =
        millis();
    }

    // Small delay between MPU6050 readings
    delay(5);
  }

  // ===================================================
  // END 10 SECOND MEASUREMENT
  // ===================================================

  unsigned long measurementEnd =
    millis();

  unsigned long elapsedTime =
    measurementEnd -
    measurementStart;

  // ===================================================
  // GET FINAL PULSE COUNT
  // ===================================================

  noInterrupts();

  unsigned long pulses =
    pulseCount;

  pulseCount = 0;

  interrupts();

  // ===================================================
  // CALCULATE VIBRATION RMS
  // ===================================================

  if (vibrationSamples > 0)
  {
    // -------------------------------------------------
    // Mean acceleration
    // -------------------------------------------------

    double meanX =
      sumX / vibrationSamples;

    double meanY =
      sumY / vibrationSamples;

    double meanZ =
      sumZ / vibrationSamples;

    // -------------------------------------------------
    // Mean square acceleration
    // -------------------------------------------------

    double meanXSquare =
      sumXSquare / vibrationSamples;

    double meanYSquare =
      sumYSquare / vibrationSamples;

    double meanZSquare =
      sumZSquare / vibrationSamples;

    // -------------------------------------------------
    // AC RMS squared
    // -------------------------------------------------

    double rmsXSquare =
      meanXSquare -
      (meanX * meanX);

    double rmsYSquare =
      meanYSquare -
      (meanY * meanY);

    double rmsZSquare =
      meanZSquare -
      (meanZ * meanZ);

    // -------------------------------------------------
    // Prevent negative floating-point values
    // -------------------------------------------------

    if (rmsXSquare < 0)
      rmsXSquare = 0;

    if (rmsYSquare < 0)
      rmsYSquare = 0;

    if (rmsZSquare < 0)
      rmsZSquare = 0;

    // -------------------------------------------------
    // Combined 3-axis vibration RMS
    // -------------------------------------------------

    vibration =
      sqrt(
        rmsXSquare +
        rmsYSquare +
        rmsZSquare
      );
  }
  else
  {
    vibration = 0.0;
  }

  // ===================================================
  // CONVERT ACCELERATION RMS TO VELOCITY RMS
  // ===================================================

  /*
     Assumed vibration frequency = 55 Hz

     mm/s RMS =
     (m/s² RMS × 1000)
     -----------------
     (2 × PI × 55)
  */

  vibration_mm_s =
    (vibration * 1000.0) /
    (2.0 * PI * 55.0);

  // ===================================================
  // RPM CALCULATION
  // ===================================================

  /*
     Measurement period = 10 seconds

     One pulse = one revolution

     RPM = revolutions / seconds × 60

     RPM = pulses / 10 × 60

     RPM = pulses × 6

     Example:

     4 pulses in 10 seconds
     = 4 × 6
     = 24 RPM

     233 pulses in 10 seconds
     = 233 × 6
     = 1398 RPM
  */

  if (targetsPerRevolution > 0)
  {
    rpm =
      ((float)pulses /
       targetsPerRevolution) *
      6.0;
  }
  else
  {
    rpm = 0.0;
  }

  // ===================================================
  // READ CURRENT
  // ===================================================

  Serial.println();
  Serial.println("Reading ACS712 current...");

  current =
    readCurrent();

  // ===================================================
  // FINAL SERIAL OUTPUT
  // ===================================================

  Serial.println();
  Serial.println("========================================");
  Serial.println("          FINAL MOTOR DATA");
  Serial.println("========================================");

  Serial.print("Measurement Time  : ");
  Serial.print(elapsedTime);
  Serial.println(" ms");

  Serial.print("Total Pulses      : ");
  Serial.println(pulses);

  Serial.print("Vibration Samples : ");
  Serial.println(vibrationSamples);

  Serial.print("Temperature       : ");
  Serial.print(temperature, 2);
  Serial.println(" °C");

  Serial.print("Vibration         : ");
  Serial.print(vibration, 2);
  Serial.println(" m/s² RMS");

  Serial.print("Vibration         : ");
  Serial.print(vibration_mm_s, 2);
  Serial.println(" mm/s RMS");

  Serial.print("Current           : ");
  Serial.print(current, 2);
  Serial.println(" A");

  Serial.print("RPM               : ");
  Serial.print(rpm, 2);
  Serial.println(" RPM");

  Serial.println("========================================");

  // ===================================================
  // SEND DATA TO THINGSPEAK
  // ===================================================

  if (WiFi.status() == WL_CONNECTED)
  {
    Serial.println();
    Serial.println("Sending data to ThingSpeak...");

    // -------------------------------------------------
    // Field 1 = Temperature
    // -------------------------------------------------

    ThingSpeak.setField(
      1,
      temperature
    );

    // -------------------------------------------------
    // Field 2 = Vibration mm/s RMS
    // -------------------------------------------------

    ThingSpeak.setField(
      2,
      vibration_mm_s
    );

    // -------------------------------------------------
    // Field 3 = Current
    // -------------------------------------------------

    ThingSpeak.setField(
      3,
      current
    );

    // -------------------------------------------------
    // Field 4 = RPM
    // -------------------------------------------------

    ThingSpeak.setField(
      4,
      rpm
    );

    // -------------------------------------------------
    // WRITE ALL FIELDS
    // -------------------------------------------------

    int responseCode =
      ThingSpeak.writeFields(
        myChannelNumber,
        myWriteAPIKey
      );

    if (responseCode == 200)
    {
      Serial.println(
        "ThingSpeak: DATA SENT SUCCESSFULLY!"
      );
    }
    else
    {
      Serial.print(
        "ThingSpeak Error Code: "
      );

      Serial.println(responseCode);
    }
  }
  else
  {
    Serial.println(
      "WiFi disconnected!"
    );

    WiFi.reconnect();
  }

  // ===================================================
  // NEXT MEASUREMENT
  // ===================================================

  Serial.println();
  Serial.println("Measurement complete.");
  Serial.println("Next 10 second measurement starting...");

  delay(1000);
}
