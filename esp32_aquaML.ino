/*
  esp32_aquaML.ino
  
  AquaEdge Water Quality Sensor Node (ESP32 Code)
  
  This sketch reads water quality sensors (pH, TDS, Turbidity, and Temperature)
  and transmits the data over Serial in CSV format to be classified by the Python ML model.
  
  Expected Output Format:
      pH,TDS,Turbidity,Temperature
      Example: "7.25,310.5,1.50,25.4"
  
  Note: This code includes both simulated data generation (default) and standard ADC formulas 
  for real sensors. Set USE_REAL_SENSORS to true when connecting hardware.
*/

#define USE_REAL_SENSORS false  // Set to true when hardware sensors are connected

// ==========================================================
// Hardware Pin Definitions (Change according to your wiring)
// ==========================================================
#define PH_PIN          34  // Analog pin for pH Sensor
#define TDS_PIN         35  // Analog pin for TDS Sensor
#define TURBIDITY_PIN   32  // Analog pin for Turbidity Sensor
#define TEMP_PIN        33  // Analog pin for Temperature Sensor (or DS18B20 data pin)

// ==========================================================
// Calibration Constants
// ==========================================================
#define ADC_REF_VOLTAGE 3.3   // ESP32 ADC reference voltage (usually 3.3V)
#define ADC_RESOLUTION  4095  // ESP32 12-bit ADC

// pH Calibration
#define PH_4_VOLTAGE    2.03  // Voltage at pH 4.0
#define PH_7_VOLTAGE    1.50  // Voltage at pH 7.0

// TDS Calibration
#define TDS_FACTOR      0.5   // Typical TDS conversion factor

void setup() {
  // Initialize serial communication at 115200 baud to match python config
  Serial.begin(115200);
  
  // Configure ADC resolution
  analogReadResolution(12);
  
  // Set up pin modes
  pinMode(PH_PIN, INPUT);
  pinMode(TDS_PIN, INPUT);
  pinMode(TURBIDITY_PIN, INPUT);
  pinMode(TEMP_PIN, INPUT);

  // Send a bootup message
  delay(1000);
  Serial.println("--- ESP32 AquaEdge Booted ---");
}

void loop() {
  float ph = 0.0;
  float tds = 0.0;
  float turbidity = 0.0;
  float temperature = 0.0;

  if (USE_REAL_SENSORS) {
    // Read raw sensor values
    ph = readPH();
    tds = readTDS(temperature); // TDS measurement is temperature-dependent
    turbidity = readTurbidity();
    temperature = readTemperature();
  } else {
    // Generate simulated sensor readings
    // Generates a mix of Safe and Unsafe states randomly
    static float time_offset = 0;
    time_offset += 0.1;

    // Use sine/cosine waves and random noise to simulate fluctuating data
    if (random(0, 100) < 30) {
      // Simulate UNSAFE events (e.g., spike in turbidity or TDS)
      ph = 7.0 + sin(time_offset) * 1.5 + (random(-20, 20) / 100.0);
      tds = 800.0 + cos(time_offset) * 300.0 + random(-10, 10);
      turbidity = 15.0 + sin(time_offset) * 10.0 + random(0, 5);
      temperature = 24.5 + sin(time_offset) * 2.0;
    } else {
      // Simulate SAFE baseline values
      ph = 7.2 + sin(time_offset) * 0.3 + (random(-10, 10) / 100.0);
      tds = 180.0 + cos(time_offset) * 40.0 + random(-5, 5);
      turbidity = 1.2 + sin(time_offset) * 0.5 + (random(0, 3) / 10.0);
      temperature = 25.0 + sin(time_offset) * 1.5;
    }
  }

  // Format and transmit over Serial in exact CSV format expected by python:
  // "pH,TDS_ppm,Turbidity_NTU,Temperature_C"
  Serial.print(ph, 2);
  Serial.print(",");
  Serial.print(tds, 1);
  Serial.print(",");
  Serial.print(turbidity, 2);
  Serial.print(",");
  Serial.println(temperature, 1);

  // Send a new sample every 2 seconds
  delay(2000);
}

// ==========================================================
// Sensor Reading Helper Functions
// ==========================================================

float readPH() {
  int rawADC = analogRead(PH_PIN);
  float voltage = (float)rawADC * ADC_REF_VOLTAGE / ADC_RESOLUTION;
  
  // Calculate pH using two-point calibration
  // Slope = (pH_7 - pH_4) / (V_pH7 - V_pH4)
  float slope = (7.0 - 4.0) / (PH_7_VOLTAGE - PH_4_VOLTAGE);
  float ph = 7.0 + (voltage - PH_7_VOLTAGE) * slope;
  
  // Constrain pH to realistic range 0-14
  return constrain(ph, 0.0, 14.0);
}

float readTDS(float currentTemp) {
  int rawADC = analogRead(TDS_PIN);
  float voltage = (float)rawADC * ADC_REF_VOLTAGE / ADC_RESOLUTION;
  
  // Temperature compensation formula:
  // CompensationCoefficient = 1.0 + 0.02 * (T - 25.0)
  float compensationCoefficient = 1.0 + 0.02 * (currentTemp - 25.0);
  float compensatedVoltage = voltage / compensationCoefficient;
  
  // Convert voltage to TDS ppm
  float tds = (133.33 * compensatedVoltage * compensatedVoltage * compensatedVoltage 
               - 255.86 * compensatedVoltage * compensatedVoltage 
               + 857.39 * compensatedVoltage) * TDS_FACTOR;
               
  return constrain(tds, 0.0, 3000.0);
}

float readTurbidity() {
  int rawADC = analogRead(TURBIDITY_PIN);
  float voltage = (float)rawADC * ADC_REF_VOLTAGE / ADC_RESOLUTION;
  
  // Convert voltage to NTU (empirical formula based on SEN0189)
  float turbidity = 0.0;
  if (voltage < 2.5) {
    turbidity = 3000.0;
  } else {
    // NTU = -1120.4 * V^2 + 5742.3 * V - 4352.9
    turbidity = -1120.4 * voltage * voltage + 5742.3 * voltage - 4352.9;
  }
  
  return constrain(turbidity, 0.0, 100.0);
}

float readTemperature() {
  // Read analog value from a NTC Thermistor or LM35 temperature sensor
  int rawADC = analogRead(TEMP_PIN);
  float voltage = (float)rawADC * ADC_REF_VOLTAGE / ADC_RESOLUTION;
  
  // Example for LM35 sensor: 10mV per degree C
  float tempC = voltage * 100.0;
  
  return constrain(tempC, 0.0, 50.0);
}
