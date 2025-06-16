#include <LiquidCrystal_I2C.h>

// set the LCD number of columns and rows
int lcdColumns = 16;
int lcdRows = 2;

// set traffic signa pins 
const int redPin = 19; 
const int yellowPin = 4; 
const int greenPin = 5;//25 


// set LCD address
LiquidCrystal_I2C lcd(0x27, lcdColumns, lcdRows);  

int countdownSeconds = 30;  // ← تبدأ العد من 30 ثانية

void setup(){
  pinMode(redPin, OUTPUT);
  pinMode(yellowPin, OUTPUT);
  pinMode(greenPin, OUTPUT);

  Serial.begin(115200); // Set baud rate
  lcd.init();
  lcd.backlight();

  

  lcd.setCursor(0, 0);
  lcd.print("Traffic Manager");

}

void loop(){
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n'); // Read incoming character
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      String action = data.substring(0, commaIndex);
      int duration = data.substring(commaIndex + 1).toInt();
      Serial.print("LED Index: ");
      Serial.print(action);
      Serial.print(" | Value: ");
      Serial.println(duration);
      set_action(action);
      set_duration(duration); 
    }
  }
  
  


}
void set_action(String command){
    if (command == "g") {
      digitalWrite(greenPin, HIGH); // Turn on LED
      digitalWrite(redPin, LOW);
      digitalWrite(yellowPin, LOW);
    } else if (command == "r"){
      digitalWrite(greenPin, LOW); // Turn off LED if any other char
      digitalWrite(redPin, HIGH);
      digitalWrite(yellowPin, LOW);
    }
    else if (command == "y"){
      digitalWrite(greenPin, LOW); // Turn off LED if any other char
      digitalWrite(redPin, LOW);
      digitalWrite(yellowPin, HIGH);
    }
}
void set_duration(int duration){
  for (int i = duration; i >= 0; i--) {
    lcd.setCursor(0, 0);     
    lcd.print("Time: ");
    lcd.print(i);
    lcd.print(" sec     ");   
    delay(1000);              
  }
  lcd.setCursor(0, 0);
  lcd.print("Traffic Manager");
}
