import os
import time
from collections import Counter, deque

import cv2
import numpy as np
from deepface import DeepFace


CAMERA_INDEX = 0
WINDOW_NAME = "Prepoznavanje facijalnih ekspresija u realnom vremenu"
ANALYZE_EVERY_N_FRAMES = 5
SMOOTHING_WINDOW = 8
MIN_CONFIDENCE = 25.0

ALLOWED_EMOTIONS = {
    "happy",
    "neutral",
    "surprise",
    "angry"
}

EMOTION_NAMES = {
    "happy": "SRECAN",
    "neutral": "NEUTRALAN",
    "surprise": "IZNENADJEN",
    "angry": "LJUT"
}

EMOTION_EMOJIS = {
    "happy": ":D",
    "neutral": ":|",
    "surprise": ":O",
    "angry": ">:("
}

EMOTION_COLORS = {
    "happy": (60, 220, 80),
    "neutral": (180, 180, 180),
    "surprise": (30, 210, 240),
    "angry": (60, 60, 230)
}

BAR_LABELS = {
    "happy": "SRECAN",
    "neutral": "NEUTRALAN",
    "surprise": "IZNENADJEN",
    "angry": "LJUT"
}


def create_reaction_card(emotion, width=520, height=520):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (24, 27, 33)

    color = EMOTION_COLORS.get(emotion, (180, 180, 180))
    emoji = EMOTION_EMOJIS.get(emotion, ":|")
    label = EMOTION_NAMES.get(emotion, "NEUTRALAN")

    cv2.circle(image, (width // 2, height // 2 - 20), 165, (35, 39, 46), -1)
    cv2.circle(image, (width // 2, height // 2 - 20), 165, color, 5)

    font_scale = 4.0
    thickness = 7

    (text_width, text_height), _ = cv2.getTextSize(
        emoji,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    emoji_x = width // 2 - text_width // 2
    emoji_y = height // 2 + text_height // 2 - 25

    cv2.putText(
        image,
        emoji,
        (emoji_x, emoji_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    (label_width, _), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        3
    )

    cv2.putText(
        image,
        label,
        (width // 2 - label_width // 2, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        color,
        3,
        cv2.LINE_AA
    )

    return image


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Kamera nije mogla da se otvori.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_number = 0
emotion_history = deque(maxlen=SMOOTHING_WINDOW)

current_emotion = "neutral"
current_confidence = 0.0

emotion_scores = {
    emotion: 0.0
    for emotion in ALLOWED_EMOTIONS
}

face_region = None
fps = 0.0
analysis_latency = 0.0
previous_time = time.perf_counter()

while True:
    success, frame = cap.read()

    if not success:
        print("Nije moguce procitati frame.")
        break

    frame_number += 1
    frame = cv2.flip(frame, 1)

    current_time = time.perf_counter()
    delta = current_time - previous_time
    previous_time = current_time

    if delta > 0:
        instant_fps = 1.0 / delta

        if fps == 0:
            fps = instant_fps
        else:
            fps = fps * 0.90 + instant_fps * 0.10

    if frame_number % ANALYZE_EVERY_N_FRAMES == 0:
        analysis_start = time.perf_counter()

        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            analysis_latency = (
                time.perf_counter() - analysis_start
            ) * 1000.0

            if results and len(results) > 0:
                result = results[0]
                raw_scores = result.get("emotion", {})

                for emotion in ALLOWED_EMOTIONS:
                    emotion_scores[emotion] = float(
                        raw_scores.get(emotion, 0.0)
                    )

                raw_emotion = max(
                    ALLOWED_EMOTIONS,
                    key=lambda emotion: emotion_scores[emotion]
                )

                raw_confidence = emotion_scores[raw_emotion]
                region = result.get("region")

                if region:
                    face_region = (
                        int(region.get("x", 0)),
                        int(region.get("y", 0)),
                        int(region.get("w", 0)),
                        int(region.get("h", 0))
                    )

                if raw_confidence >= MIN_CONFIDENCE:
                    emotion_history.append(raw_emotion)

                if emotion_history:
                    counts = Counter(emotion_history)
                    current_emotion = counts.most_common(1)[0][0]
                    current_confidence = emotion_scores.get(
                        current_emotion,
                        0.0
                    )

        except Exception as error:
            print("Upozorenje pri analizi:", error)

    camera_display = cv2.resize(frame, (700, 520))

    if face_region is not None:
        x, y, w, h = face_region

        scale_x = 700 / frame.shape[1]
        scale_y = 520 / frame.shape[0]

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))

        cv2.rectangle(
            camera_display,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        label = f"{EMOTION_NAMES[current_emotion]} {current_confidence:.1f}%"

        cv2.putText(
            camera_display,
            label,
            (x, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA
        )

    reaction = create_reaction_card(current_emotion, 520, 520)
    display = np.hstack((camera_display, reaction))

    hud_height = 150

    final_display = np.zeros(
        (520 + hud_height, display.shape[1], 3),
        dtype=np.uint8
    )

    final_display[:] = (17, 20, 25)
    final_display[hud_height:, :] = display

    cv2.putText(
        final_display,
        "PREPOZNAVANJE FACIJALNIH EKSPRESIJA U REALNOM VREMENU",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    color = EMOTION_COLORS.get(current_emotion, (255, 255, 255))

    cv2.putText(
        final_display,
        f"Ekspresija: {EMOTION_NAMES[current_emotion]}",
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        final_display,
        f"Pouzdanost: {current_confidence:.1f}%",
        (315, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        final_display,
        f"FPS: {fps:.1f}",
        (590, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (70, 230, 150),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        final_display,
        f"AI kasnjenje: {analysis_latency:.0f} ms",
        (750, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (70, 210, 240),
        2,
        cv2.LINE_AA
    )

    bar_names = [
        "happy",
        "neutral",
        "surprise",
        "angry"
    ]

    start_x = 25
    start_y = 110

    for index, emotion in enumerate(bar_names):
        score = emotion_scores[emotion]
        x = start_x + index * 290
        label = f"{BAR_LABELS[emotion]} {score:.0f}%"

        cv2.putText(
            final_display,
            label,
            (x, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 210, 220),
            1,
            cv2.LINE_AA
        )

        bar_width = 150

        cv2.rectangle(
            final_display,
            (x, start_y + 10),
            (x + bar_width, start_y + 20),
            (50, 55, 62),
            -1
        )

        fill_width = int(
            bar_width * min(score, 100.0) / 100.0
        )

        cv2.rectangle(
            final_display,
            (x, start_y + 10),
            (x + fill_width, start_y + 20),
            EMOTION_COLORS[emotion],
            -1
        )

    cv2.putText(
        final_display,
        "KAMERA UZIVO",
        (20, hud_height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        final_display,
        "VIZUELNA REAKCIJA",
        (720, hud_height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        final_display,
        "Q - Izlaz    S - Snimak ekrana",
        (970, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 185, 195),
        1,
        cv2.LINE_AA
    )

    cv2.imshow(WINDOW_NAME, final_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):
        os.makedirs("screenshots", exist_ok=True)

        filename = os.path.join(
            "screenshots",
            f"ekspresija_{int(time.time())}.jpg"
        )

        cv2.imwrite(filename, final_display)
        print("Snimak sacuvan:", filename)

cap.release()
cv2.destroyAllWindows()

print("Program zavrsen.")
