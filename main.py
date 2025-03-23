Step 1: Install Required Libraries
!pip install torch torchvision torchaudio transformers ffmpeg-python opencv-python moviepy ultralytics speechbrain
!apt-get install -y ffmpeg



Step 2: Extract Frames from the Video
import cv2
import os

video_path = "/content/WhatsApp Video 2025-03-23 at 5.08.07 PM.mp4"  # Update with your video file path
frame_dir = "/content/frames"

os.makedirs(frame_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 5  # Extract 1 frame per 5 seconds
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % (frame_rate * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
        frame_filename = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()




Step 3: Detect Cricketing Events Using YOLOv8

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Smallest YOLOv8 model, can be fine-tuned for cricket
event_dict = {0: "Batsman", 1: "Bowler", 2: "Wicketkeeper", 3: "Six", 4: "Four", 5: "Catch", 6: "Run-out", 7: "Bowled"}

events = []

for frame_file in sorted(os.listdir(frame_dir)):
    frame_path = os.path.join(frame_dir, frame_file)
    results = model(frame_path)

    for result in results:
        for det in result.boxes:
            class_id = int(det.cls[0].item())
            if class_id in event_dict:
                events.append(event_dict[class_id])

print("Detected Cricketing Events:", events)




Step 4: Generate AI Commentary

# Define structured cricket commentary templates
event_to_commentary = {
    "Four": [
        "That's a classic boundary! The batsman finds the gap beautifully.",
        "Brilliant shot! The ball races to the fence for four.",
        "Perfectly timed! The batsman places it past the fielder for a four."
    ],
    "Six": [
        "Massive hit! The ball sails over the ropes for a six.",
        "That's out of the park! The batsman sends it deep into the stands.",
        "Huge six! The bowler looks under pressure now."
    ],
    "Bowled": [
        "Clean bowled! The batsman is completely beaten.",
        "What a delivery! The stumps are shattered.",
        "That's a peach of a delivery! The batsman has to walk back."
    ],
    "Catch": [
        "Caught out! The fielder takes a sharp catch.",
        "Excellent fielding! The batsman is dismissed.",
        "What a grab! The bowler celebrates as the batsman walks back."
    ],
    "Single": [
        "Quick single taken! The batsmen run well between the wickets.",
        "Smart running! The batsman taps it and takes one.",
        "Good awareness! The batsmen rotate the strike."
    ],
    "Double": [
        "Well run! The batsmen push for two.",
        "Good placement! They manage a comfortable two runs.",
        "Excellent running! The batsmen convert one into two."
    ],
}

import random

# Example detected events from video
detected_events = ["Four", "Six", "Bowled"]  # Replace with actual detected events

generated_commentary = []

for event in detected_events:
    if event in event_to_commentary:
        commentary = random.choice(event_to_commentary[event])  # Select a random predefined commentary
        generated_commentary.append(commentary)

print("AI-Generated Commentary:", generated_commentary)




from transformers import pipeline

commentary_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

final_commentary = []

for commentary in generated_commentary:
    prompt = f"A cricket commentator is describing a match. The commentary should be natural and exciting. Here's the original: '{commentary}'\n\nThe improved version is:"
    enhanced_commentary = commentary_model(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    final_commentary.append(enhanced_commentary)

print("Final AI-Generated Commentary:", final_commentary)






Step 5: Convert Text Commentary to Speech (TTS)

from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

output_audio_files = []

for idx, text in enumerate(generated_commentary):
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)

    audio_filename = f"/content/commentary_{idx}.wav"
    torchaudio.save(audio_filename, waveforms.squeeze(1), 22050)
    output_audio_files.append(audio_filename)

print("Generated Audio Files:", output_audio_files)
