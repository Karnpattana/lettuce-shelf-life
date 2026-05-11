# Shared Lab* thresholds — ใช้ค่าเดียวกันทั้งใน color.py และ shape.py (กัน threshold drift)
# อ้างอิง: pixel classification ตาม Kader OVQ visual criteria

GREEN_LAB_RANGE = {"L_min": 30, "a_max": -5}
YELLOW_LAB_RANGE = {"a_min": -5, "a_max": 5, "b_min": 20}
BROWN_LAB_RANGE = {"a_min": 5, "b_min": 10, "L_max": 50}
