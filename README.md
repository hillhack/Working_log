## Employee Monitoring System using OpenCV & MediaPipe

This project is a real-time **employee activity monitoring system** that uses **OpenCV** and **MediaPipe** to determine whether a person is working or distracted based on:

* **Eye status** (closed for > 5 minutes)
* **Face presence** (absent for > 2 minutes)
* **Mouth activity** (talking continuously for long or talking mode > 4)

 The system logs sessions to a CSV file to track whether the person was actively working or distracted.

###  Features

* **Eye Tracking** – Detects if eyes are closed using EAR (Eye Aspect Ratio)
* **Face Presence Detection** – Logs absence if face not detected for over 2 minutes
* **Mouth Movement Tracking** – Detects if person is talking continuously (based on MAR - Mouth Aspect Ratio)
 **Logs Output** – CSV file records date, start/end time, duration, eye status, and talking status

###  How It Works

The app uses **MediaPipe FaceMesh** to detect facial landmarks:

1. **Eye Aspect Ratio (EAR)** is used to detect closed eyes.
2. **Mouth Aspect Ratio (MAR)** + variation is used to determine if the person is continuously talking.
3. **MediaPipe Face Detection** identifies whether the face is visible (i.e., present).

The logic then logs the person as:

* `Not Working` if:

  * Eyes are closed for more than **5 minutes**, or
  * Face is **not detected** for more than **2 minutes**, or
  * Talking mode (mouth active) is **4 times or more** in a session


### 🚀 How to Run

1. **Clone this repo**

   ```bash
   git clone https://github.com/hillhack/Working_log.git
   ```

2. **Install dependencies**

   ```bash
   pip install opencv-python mediapipe
   ```

3. **Start the App**

   ```bash
   python main.py
   ```

### Output Log

Logs are saved in `data/presence_log.csv` like this:

| Date       | Start Time | End Time | Duration (s) | Eye Status  | Talking     |
| ---------- | ---------- | -------- | ------------ | ----------- | ----------- |
| 2025-07-08 | 10:12:00   | 10:15:30 | 210          | Eyes Open   | Talking     |
| 2025-07-08 | 11:02:12   | 11:07:14 | 302          | Eyes Closed | Not Talking |

---

### 📌 Credits

Developed by :
👩‍🎓 Data Science Student @ IIT Madras
🔗 Email: [2jyotihill@gmail.com](mailto:2jyotihill@gmail.com)
