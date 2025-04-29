# test.py
from model import ShipDetector

def test_detector():
    detector = ShipDetector()
    assert detector.model is not None
    print("Тест пройден успешно!")

if __name__ == "__main__":
    test_detector()