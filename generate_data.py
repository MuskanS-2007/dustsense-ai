"""
Data generator for DustSentinel: Realistic Synthetic Sensor Data
Tech For Earth Hackathon 2026 - Construction Fugitive Dust Monitoring
"""
import csv
import math
import random
from datetime import datetime, timedelta

def generate_sample_dataset():
    random.seed(42)  # Deterministic seed for reproducible baseline
    
    start_time = datetime(2026, 10, 13, 6, 0, 0)
    intervals = 96  # 24 hours at 15-minute intervals
    
    zones = ["Excavation", "Haul Road", "Material Storage", "Loading Point"]
    
    rows = []
    
    for i in range(intervals):
        current_time = start_time + timedelta(minutes=15 * i)
        hour = current_time.hour + current_time.minute / 60.0
        
        # Diurnal weather dynamics
        # Temperature peaks around 14:00 (34C), lowest at 06:00 (22C)
        temp_cycle = math.sin((hour - 8.0) * math.pi / 12.0)
        temperature = round(28.0 + 6.0 * temp_cycle + random.uniform(-0.5, 0.5), 1)
        
        # Humidity is inversely correlated with temperature
        humidity = round(max(20.0, min(85.0, 52.0 - 22.0 * temp_cycle + random.uniform(-2.0, 2.0))), 1)
        
        # Wind speed increases midday
        base_wind = 9.0 + 8.0 * max(0.0, temp_cycle) + random.uniform(-1.5, 2.0)
        # Afternoon gust event between 13:00 and 15:30
        if 13.0 <= hour <= 15.5:
            base_wind += random.uniform(5.0, 11.0)
        wind_speed = round(max(3.0, base_wind), 1)
        
        # Wind direction: prevailing West-Southwest (~245 deg) with slight drift
        wind_deg = round((245.0 + 20.0 * math.sin(hour * math.pi / 6.0) + random.uniform(-10.0, 10.0)) % 360, 1)
        
        for zone in zones:
            # Baseline background ambient particulate
            bg_pm25 = 18.0 + random.uniform(-2.0, 3.0)
            bg_pm10 = 35.0 + random.uniform(-4.0, 5.0)
            
            activity = "Low"
            dust_pm10 = 0.0
            dust_pm25 = 0.0
            
            # Site working hours: 07:30 to 17:30
            is_work_hours = 7.5 <= hour <= 17.5
            
            if zone == "Excavation":
                if is_work_hours:
                    activity = "High" if (8.5 <= hour <= 12.0 or 13.5 <= hour <= 16.5) else "Moderate"
                    # Earthmoving produces heavy coarse dust (high PM10, moderate PM2.5)
                    intensity = 2.2 if activity == "High" else 1.2
                    dust_pm10 = random.uniform(70.0, 140.0) * intensity
                    dust_pm25 = dust_pm10 * random.uniform(0.18, 0.28) # Crustal dust has low PM2.5/PM10 ratio
                else:
                    activity = "Low"
                    dust_pm10 = random.uniform(5.0, 15.0)
                    dust_pm25 = random.uniform(2.0, 6.0)
                    
            elif zone == "Haul Road":
                if is_work_hours:
                    activity = "High" if (9.0 <= hour <= 16.0) else "Moderate"
                    # Episodic truck passes cause rapid spikes
                    truck_pass = random.random() < (0.65 if activity == "High" else 0.3)
                    if truck_pass:
                        dust_pm10 = random.uniform(130.0, 260.0)
                        dust_pm25 = dust_pm10 * random.uniform(0.20, 0.32)
                    else:
                        dust_pm10 = random.uniform(30.0, 65.0)
                        dust_pm25 = dust_pm10 * 0.25
                else:
                    activity = "None"
                    dust_pm10 = random.uniform(2.0, 10.0)
                    dust_pm25 = random.uniform(1.0, 4.0)
                    
            elif zone == "Material Storage":
                # Wind erosion dependent: stockpiles emit dust when wind is high and humidity is low
                wind_dry_factor = max(0.0, (wind_speed - 14.0) / 10.0) * max(0.0, (45.0 - humidity) / 25.0)
                if wind_dry_factor > 0:
                    activity = "Moderate" if is_work_hours else "Low"
                    dust_pm10 = 50.0 + (wind_dry_factor * random.uniform(110.0, 190.0))
                    dust_pm25 = dust_pm10 * random.uniform(0.15, 0.24)
                elif is_work_hours:
                    activity = "Moderate"
                    dust_pm10 = random.uniform(25.0, 55.0)
                    dust_pm25 = random.uniform(8.0, 18.0)
                else:
                    activity = "None"
                    dust_pm10 = random.uniform(3.0, 12.0)
                    dust_pm25 = random.uniform(1.0, 5.0)
                    
            elif zone == "Loading Point":
                # Transfer points have cyclical loading operations
                if is_work_hours and ((9.0 <= hour <= 11.5) or (13.5 <= hour <= 16.0)):
                    activity = "High"
                    dust_pm10 = random.uniform(90.0, 185.0)
                    dust_pm25 = dust_pm10 * random.uniform(0.25, 0.38)
                elif is_work_hours:
                    activity = "Moderate"
                    dust_pm10 = random.uniform(35.0, 70.0)
                    dust_pm25 = dust_pm10 * 0.30
                else:
                    activity = "None"
                    dust_pm10 = random.uniform(4.0, 12.0)
                    dust_pm25 = random.uniform(2.0, 5.0)
            
            pm10_final = round(bg_pm10 + dust_pm10, 1)
            pm25_final = round(bg_pm25 + dust_pm25, 1)
            
            rows.append({
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "zone": zone,
                "pm2_5": pm25_final,
                "pm10": pm10_final,
                "wind_speed": wind_speed,
                "wind_deg": wind_deg,
                "humidity": humidity,
                "temperature": temperature,
                "activity_level": activity
            })
            
    with open("sample_sensor_data.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "zone", "pm2_5", "pm10", "wind_speed", "wind_deg", "humidity", "temperature", "activity_level"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} rows into sample_sensor_data.csv successfully!")

if __name__ == "__main__":
    generate_sample_dataset()
