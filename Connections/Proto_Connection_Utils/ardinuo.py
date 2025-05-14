import serial
import time

class ardinuo:
    def __init__(self,lane_id,port,baudrate=115200):
        self.lane_id=lane_id
        self.port=port
        self.baudrate=baudrate
        self.set_Esp32()

    def set_Esp32(self):
        # Replace with your actual COM port. For example, '/dev/ttyUSB0' on Linux or 'COM3' on Windows.
        self.arduino = serial.Serial(self.port, self.baudrate)
        time.sleep(2)  # Give Arduino time to reset
        

    def send_command(self, light, duration):
        # Create the command string
        command = f"{light},{duration}\n"  # Format: light,duration
        # Send the command to the Arduino
        self.arduino.write(command.encode())
        time.sleep(0.5)  # Wait for a short period to ensure the command is sent
        

    def close_arduino(self):
        # Close the serial connection
        self.arduino.close()



