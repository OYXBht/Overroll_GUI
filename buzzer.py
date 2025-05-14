#!/usr/bin/env python3
# coding=utf-8

import time
import sys

sys.path.append('/home/pi/TurboPi')

try:
    import HiwonderSDK.Board as Board
    from HiwonderSDK.Sonar import Sonar
    HW_AVAILABLE = True
except ImportError as e:
    HW_AVAILABLE = False
    raise e

sonar = Sonar() if HW_AVAILABLE else None


def check_distance_and_buzz(threshold=10.0, delay=0.2):
    """
    检查超声波距离并控制蜂鸣器。
    如果距离小于 threshold（厘米），则响铃；否则关闭。
    """
    if not HW_AVAILABLE:
        print("[WARN] HiwonderSDK 未找到，蜂鸣器不可用。")
        return

    try:
        dist_cm = sonar.getDistance() / 10.0
        print(f"[Buzzer] 距离: {dist_cm:.1f} cm")
        if dist_cm < threshold:
            Board.setBuzzer(0)  # 开启蜂鸣器
        else:
            Board.setBuzzer(1)  # 关闭蜂鸣器
        time.sleep(delay)
        Board.setBuzzer(1)
    except Exception as e:
        print(f"[Buzzer Error] {e}")
        Board.setBuzzer(1)


if __name__ == '__main__':
    check_distance_and_buzz()
