#!/usr/bin/env python3
# coding=utf-8

import sys
sys.path.append('/home/pi/TurboPi')
from HiwonderSDK.Sonar import Sonar
import time


def main():
    sonar = Sonar()
    print("Ultrasonic test started. Ctrl+C to exit.")
    try:
        while True:
            dist_cm = sonar.getDistance() / 10.0
            print(f"Distance: {dist_cm:.1f} cm")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
