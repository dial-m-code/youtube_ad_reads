"""
Konvertiert Json3-Dateien und Pickles in ein einheitliches Parquet-Dataset
"""

import os
import pickle
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

VIDEO_DIR = Path("Projektarbeit_YT/raw_data/video_meta")
SUBTITLE_DIR = Path("Projektarbeit_YT/raw_data/subs")
OUTPUT_DIR = Path("converted_dataset")
CACHE_FILE = OUTPUT_DIR / "conversion_checkpoint.json"
BATCH_SIZE = 100000


def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Fehler beim Laden der PKL-Datei: {file_path}: {e}")
        return None

def load_subtitle(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'events' not in data:
            return []
        return [
            {
                "start_ms": event.get("tStartMs"),
                "duration_ms": event.get("dDurationMs"),
                "text": ''.join(seg.get("utf8", "") for seg in event.get("segs", []))
            }
            for event in data["events"] if "segs" in event
        ]
    except Exception as e:
        print(f"Fehler beim Laden der JSON3-Datei: {file_path}: {e}")
        return []

def load_checkpoint():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(processed_ids):
    with open(CACHE_FILE, 'w') as f:
        json.dump(list(processed_ids), f)

def convert_and_save_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_video_files = sorted(VIDEO_DIR.glob("*.pkl"))
    processed_ids = load_checkpoint()
    
    # Indexiere alle englischen Untertitel-Dateien
    subtitle_map = {
        f.name.split(".")[0]: f
        for f in SUBTITLE_DIR.glob("*.en.json3")
    }

    current_batch = []
    batch_index = 0
    total_files = len(all_video_files)

    with tqdm(total=total_files, desc="Konvertiere Dataset") as pbar:
        for video_file in all_video_files:
            video_id = video_file.stem
            if video_id in processed_ids:
                pbar.update(1)
                continue

            video_data = load_pickle(video_file)
            if video_data is None:
                pbar.update(1)
                continue

            subtitle_path = subtitle_map.get(video_id)
            subtitles = load_subtitle(subtitle_path) if subtitle_path else []

            # video_id, video_title, published, description, channel_id
            # default_audio_lang, topics, snippet, statistics, sponsor_data, subtitles
            current_batch.append({
                "id": video_id,
                "channel_id": video_data.get("snippet", {}).get("channelId"),
                "channel_title": video_data.get("snippet", {}).get("channelTitle"),
                "title": video_data.get("snippet", {}).get("title"),
                "description": video_data.get("snippet", {}).get("description"),
                "published_at": video_data.get("snippet", {}).get("publishedAt"),
                "view_count": video_data.get("statistics", {}).get("viewCount"),
                "default_language": video_data.get("snippet", {}).get("defaultLanguage", video_data.get("snippet", {}).get("defaultAudioLanguage")),
                "topic_categories": video_data.get("topicDetails", {}).get("topicCategories"),
                "subtitles": subtitles,
                "snippet": video_data.get("snippet"),
                "statistics": video_data.get("statistics")
            })

            processed_ids.add(video_id)
            pbar.update(1)

            if len(current_batch) >= BATCH_SIZE:
                save_batch(current_batch, batch_index)
                batch_index += 1
                current_batch.clear()
                save_checkpoint(processed_ids)

        if current_batch:
            save_batch(current_batch, batch_index)
            save_checkpoint(processed_ids)

def save_batch(batch_data, index, kind="videos_batch_"):
    df = pd.DataFrame(batch_data)
    output_path = OUTPUT_DIR / f"{kind}{index:04d}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"[INFO] Batch {index} gespeichert: {output_path}")

def create_channel_dataset(channel_data_dir:Path, channel_topic_dir:Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    channel_dataset = []
    all_channels = channel_data_dir.glob("*.pkl")
    topic_map = {
        f.name.split(".")[0]: f
        for f in channel_topic_dir.glob("*.pkl")
    }
    for channel in tqdm(all_channels):
        #print(channel)
        channel_id=channel.stem
        channel_data = load_pickle(channel)
        if channel_data is None:
            continue
        topic_path = topic_map.get(channel_id)
        topics = load_pickle(topic_path) if topic_path else {}
        channel_dataset.append({
            "id": channel_data.get("id"),
            "channel_title": channel_data.get("snippet", {}).get("title"),
            "channel_description": channel_data.get("snippet", {}).get("description"),
            "channel_handle": channel_data.get("snippet", {}).get("customUrl"),
            "channel_country": channel_data.get("snippet", {}).get("country"),
            "channel_topic_ids": topics.get("topicDetails", {}).get("topicIds"),
            "channel_topic_categories": topics.get("topicDetails", {}).get("topicCategories"),
            "channel_snippet": channel_data.get("snippet"),
            "channel_statistics": channel_data.get("statistics")
        })
    save_batch(channel_dataset, 0, "channel_data_")



if __name__ == "__main__":
    convert_and_save_dataset()
    #create_channel_dataset(Path("Projektarbeit_YT/raw_data/channel_meta"), Path("Projektarbeit_YT/raw_data/topic_channel"))