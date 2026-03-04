"""
Lädt den Datensatz, führt Filterung, NER und NER-Ranking aus.
Validierung und manuelle Listen werden genutzt, um NER-Ergebnisse zu verbessern.
"""

import pickle
#import json
#from xml.etree.ElementInclude import include
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path
from datasets import Dataset
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter

import unicodedata
import re

import random

# nltk.download("punkt")

# DATA-PATHS
VIDEO_DATA = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/converted_dataset"
CHANNEL_DATA = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/converted_dataset"
SPONSOR_DATA = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/converted_dataset/sponsor_times.pkl"

NER_RESULTS = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/scripts"

EXCLUDE_ENT_PATH = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/scripts/exclude_ent.txt"
MATCH_DICT_PATH = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/scripts/match_dict.csv"

KNOWN_BRANDS_PATH = "/Users/lmeyer/Programmierung DH/Projektarbeit_YT/scripts/known_sponsors.txt"

# Known Brands RE erstellen
with open(KNOWN_BRANDS_PATH, "r") as f:
    KNOWN_BRANDS = f.read().splitlines()
KNOWN_BRANDS_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, KNOWN_BRANDS)) + r')\b', flags=re.IGNORECASE)


# Exclude erstellen
with open(EXCLUDE_ENT_PATH, "r") as f:
    EXCLUDE_ENT = set(f.read().splitlines())

# Match-Dict erstellen
match_df = pd.read_csv(MATCH_DICT_PATH)
MATCH_DICT = dict(zip(match_df["key"],match_df["match"]))
del match_df

# NLP-Setup

#tokenizer = AutoTokenizer.from_pretrained("jayant-yadav/roberta-base-multinerd")
#model = AutoModelForTokenClassification.from_pretrained("jayant-yadav/roberta-base-multinerd")
#ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device="mps")
ner_pipeline = None

def main():
    #dataset=load_dataset()
    #ner_res = dataset.sponsors.extract_org_entities_from_all_videos_ds()
    
    print()


    print()

# Klassen

class YouTubeAdDataset:
    def __init__(self, df_videos: pd.DataFrame, df_channels: pd.DataFrame, 
                 df_sponsors: pd.DataFrame, ner_results: pd.DataFrame):
        self.videos = VideoView(df_videos)
        self.channels = ChannelView(df_channels)
        self.sponsors = SponsorView(df_sponsors, self.videos, ner_results, self.channels)

    def __repr__(self):
        return f"YouTubeAdDataset with {len(self.videos)} videos"
    
    def create_validation_sample(self, n: int = 150, exclude_txt: str = '/Users/lmeyer/Programmierung DH/Projektarbeit_YT/scripts/validated_ids.txt'):
        with open(exclude_txt, "r") as f:
            exclude = set(f.read().splitlines())
        
        yt_link_format = "http://youtube.com/watch?v="

        sponsors_labeled = self.sponsors.label_sponsors(self.sponsors.ner_results)
        sponsors_labeled = sponsors_labeled[~sponsors_labeled["video_id"].isin(exclude)]
        sponsor_sample = sponsors_labeled.sample(n)

        sponsor_df = self.sponsors.sponsor_df
        sponsor_df = sponsor_df[
            (sponsor_df["category"] == "sponsor") & 
            ((sponsor_df["votes"] >= 0 ) | (sponsor_df["locked"] == 1))
        ]


        validation_table = []
        for _, row in sponsor_sample.iterrows():
            video_id = row["video_id"]

            sponsor_segs = self.sponsors.overlap_filter(sponsor_df[sponsor_df["videoID"] == video_id])

            if sponsor_segs.empty:
                continue

            if "locked" in sponsor_segs.columns and sponsor_segs["locked"].eq(1).any():
                sponsor_segs = sponsor_segs[sponsor_segs["locked"] == 1]
                sponsor_segs = sponsor_segs.head(3)
            else:
                sponsor_segs = sponsor_segs.head(3)

            sponsor_segs = sponsor_segs.sort_values("startTime").reset_index(drop=True)

            current_video = {"video_id": video_id}
            for i, row_s in sponsor_segs.iterrows():
                sponsor_start = int(row_s["startTime"])
                current_video[f"link_to_sponsor_{i + 1}"] = f"{yt_link_format}{video_id}&t={sponsor_start}"

            current_video["sponsor_candidates"] = row["sponsor_ents"]
            current_video["sponsor"] = row["sponsor"]
            current_video["real_sponsor"] = ""
            current_video["in_candidates"] = ""

            validation_table.append(current_video)

        return pd.DataFrame(validation_table)



class VideoView:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_top_videos(self, n=10):
        return self.df.sort_values(by='view_count', ascending=False).head(n)

    def get_by_id(self, video_id: str):
        return self.df[self.df['id'] == video_id]

    def filter_by_language(self, lang: str = 'en'):
        #return self.df[self.df['default_language'].str.startswith(lang, na=False)]
        return self.df[(self.df["default_language"]==lang)&
                                 (self.df["subtitles"].apply(lambda x: len(x) > 0 ))]
    
    def get_channel_name_by_id(self, video_id: str):
        return self.df[self.df['id'] == video_id]["channel_title"].to_string(index=False)

class ChannelView:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        #self.channel_names_normed = {normalize_name(ch) for ch in self.df["channel_title"]}

    def get_unique_channels(self):
        return self.df['channelId'].nunique()

    def get_channel_stats(self):
        return self.df.groupby('channelId').agg({
            'id': 'count',
            'viewCount': 'sum',
            'likeCount': 'sum'
        }).rename(columns={'id': 'videoCount'})

    def get_by_channel_id(self, channel_id: str):
        return self.df[self.df['channelId'] == channel_id]
    
class SponsorView:
    def __init__(self, sponsor_df: pd.DataFrame, video_view: VideoView, ner_results: pd.DataFrame, channel_view: ChannelView):
        self.sponsor_df = sponsor_df
        self.video_view = video_view
        self.channel_view = channel_view
        self.video_df = video_view.df
        self.ner_results = ner_results
        self.video_ids_en = set(self.video_view.filter_by_language()["id"])

    def get_subtitles_within_sponsor_segments(self, video_id: str) -> list[dict]:
        """Gibt alle Untertitel-Segmente eines Videos zurück, die innerhalb der Sponsor-Zeiträume liegen."""
        video_row = self.video_df[self.video_df["id"] == video_id]
        if video_row.empty:
            return []

        subtitles = video_row.iloc[0]["subtitles"]
        sponsors = self.sponsor_df[(self.sponsor_df["videoID"] == video_id) & 
                                   (self.sponsor_df["category"] == "sponsor") &
                                   (self.sponsor_df["votes"] >= 0 )]
        sponsors = self.overlap_filter(sponsors)
        matches = []
        for _, s_row in sponsors.iterrows():
            s_start, s_end = seconds_to_ms(float(s_row["startTime"])), seconds_to_ms(float(s_row["endTime"]))
            for subtitle in subtitles:
                sub_start = int(subtitle.get("start_ms", 0))
                if subtitle.get("duration_ms", None) is None:
                    continue
                sub_end = sub_start + int(subtitle.get("duration_ms", 0))
                if sub_end > s_start and sub_start < s_end:
                    matches.append(subtitle)
        return matches

    def get_sponsor_text(self, video_id: str) -> str:
        """Gibt den zusammenhängenden Text aller Untertitel-Segmente innerhalb der Sponsor-Zeiträume zurück."""
        segments = self.get_subtitles_within_sponsor_segments(video_id)
        texts = []
        for segment in segments:
            text = segment.get("text", "").strip()
            text = text.replace("\n", " ").replace("\r", " ")
            if text:
                texts.append(text)
        return " ".join(texts)
    
    def extract_textfrom_sub(self, sub_out: str) -> str:
        """Gibt den zusammenhängenden Text aller Untertitel-Segmente zurück."""
        segments = sub_out
        texts = []
        for segment in segments:
            text = segment.get("text", "").strip()
            text = text.replace("\n", " ").replace("\r", " ")
            if text:
                texts.append(text)
        return " ".join(texts)
    
    def extract_org_entities_from_all_videos(self) -> dict:
        results = {}
        filtered_videos = self.video_view.filter_by_language()
        #filtered_videos = filtered_videos[filtered_videos["subtitles"].notnull()]
        #filtered_videos = filtered_videos.head(2000)
        with tqdm(total=len(filtered_videos), desc="NER") as pbar:
            for _, row in filtered_videos.iterrows():
                video_id = row["id"]
                text = self.get_sponsor_text(video_id)
                if not text.strip():
                    pbar.update(1)
                    continue
                ner_results = ner_pipeline(text)
                entities = []
                for ent in ner_results:
                    #print(ent)
                    if ent.get("entity_group") == "ORG":
                        word = ent["word"].strip()
                        entities.append(word)
                if entities:
                    results[video_id] = entities
                pbar.update(1)
        return results

    def extract_org_entities_from_all_videos_ds(self, max_videos = None, min_score: float = 0.85) -> pd.DataFrame:
        # Videos filtern
        filtered_videos = self.video_view.filter_by_language()
        #filtered_videos = filtered_videos[filtered_videos["subtitles"].notnull()].copy()
        if max_videos:
            filtered_videos = filtered_videos.head(max_videos)

        # Sponsor-Timestamps vorbereiten
        sponsor_df = self.sponsor_df
        sponsor_df = sponsor_df[
            (sponsor_df["category"] == "sponsor") & 
            ((sponsor_df["votes"] >= 0 ) | (sponsor_df["locked"] == 1)) &
            (sponsor_df["videoID"].isin(self.video_ids_en))
        ]

        # Overlaps filtern
        sponsor_df = sponsor_df.groupby("videoID", group_keys=False).apply(self.overlap_filter)

        sponsor_map = {}
        for _, row in sponsor_df.iterrows():
            sponsor_map.setdefault(row["videoID"], []).append((
                seconds_to_ms(row["startTime"]), 
                seconds_to_ms(row["endTime"])))

        # Sponsor-Text extrahieren
        def extract_text(row):
            video_id = row["id"]
            subtitles = row["subtitles"]
            timestamps = sponsor_map.get(video_id, [])
            if not timestamps:
                return ""

            segments = []
            for s_start, s_end in timestamps:
                for seg in subtitles:
                    if seg.get("duration_ms") is None:
                        continue
                    sub_start = seg.get("start_ms", 0)
                    sub_end = sub_start + seg["duration_ms"]
                    if sub_end > s_start and sub_start < s_end:
                        text = seg.get("text", "").replace("\n", " ").replace("\xa0", " ").strip()
                        if text:
                            segments.append(text)
            return " ".join(segments)

        tqdm.pandas(desc="Sponsor-Texte extrahieren")
        filtered_videos["sponsor_text"] = filtered_videos.progress_apply(extract_text, axis=1)
        filtered_videos = filtered_videos[filtered_videos["sponsor_text"].str.strip().astype(bool)]

        # Untertitel mit Description-Text zusammenführen
        filtered_videos["combined_text"] = (
            filtered_videos["sponsor_text"].fillna("") + 
            " " + 
            filtered_videos["description"].fillna("")
            )

        # HuggingFace Dataset erstellen
        ds = Dataset.from_pandas(filtered_videos[["id", "combined_text"]])

        # NER-Anwendung
        def extract_ner(example):
            entities = []
            scores = []
            for ent in ner_pipeline(example["combined_text"]):
                if ent.get("entity_group") == "ORG":
                    word = ent["word"].strip()
                    entities.append(word)
                    scores.append(float(ent["score"]))
            return {"org_entities": entities, "org_scores": scores}

        ds = ds.map(extract_ner, batched=False, desc="NER anwenden")

        # In DataFrame konvertieren
        df_result = pd.DataFrame({
            "video_id": ds["id"],
            "org_entities": ds["org_entities"],
            "org_scores": ds["org_scores"]
        })

        # Optional: nur Zeilen mit mindestens 1 Entity
        df_result = df_result[df_result["org_entities"].str.len() > 0].reset_index(drop=True)
        return df_result

    def rank_sponsor_entities(self, ner_df: pd.DataFrame, top_k: int = 5, min_len: int = 3, max_len: int = 30) -> pd.DataFrame:
        """
        Normalisiert und rankt ORG-Entitäten aus NER-Ergebnissen.

        Args:
            ner_df (pd.DataFrame): DataFrame mit Spalten 'id' und 'org_entities'
            top_k (int): Anzahl der meistgenannten Entitäten pro Video
            min_len (int): Minimale Länge einer Entität nach Normalisierung

        Returns:
            pd.DataFrame: DataFrame mit 'video_id', 'top_sponsors' (Liste), 'top_sponsor' (String)
        """
        # Mapping hinzufügen
        def normalize(entity: str) -> str:
            entity = entity.strip().lower().replace("\n", " ")
            entity = unicodedata.normalize("NFKC", entity)
            return " ".join(entity.split())  # entferne doppelte Leerzeichen

        results = []

        for _, row in ner_df.iterrows():
            video_id = row["video_id"]
            entities = row.get("org_entities", [])
            normed = [normalize(e) for e in entities if isinstance(e, str) and len(normalize(e)) >= min_len
                      and len(normalize(e)) < max_len]

            if not normed:
                continue

            counter = Counter(normed)
            top_entities = [ent for ent, _ in counter.most_common(top_k)]

            results.append({
                "video_id": video_id,
                "top_sponsors": top_entities,
                "top_sponsor": top_entities[0] if top_entities else None
            })

        return pd.DataFrame(results)

    def overlap_filter(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group['startTime'] = group['startTime'].astype(float)
        group['endTime'] = group['endTime'].astype(float)
        group['duration'] = group['endTime'] - group['startTime']

        # Sortieren nach startTime
        group.sort_values(by=['startTime', 'locked', 'votes', 'views', 'duration'], ascending=[True, False, False, False, False], inplace=True)

        selected_rows = []
        last_end = -np.inf

        for idx, row in group.iterrows():
            if row['startTime'] >= last_end:
                selected_rows.append(idx)
                last_end = row['endTime']

        return group.loc[selected_rows].drop(columns='duration')
    
    def get_subtitles_within_sponsor_segments_windowed(self, video_id: str, position: str, window_width: int = 7) -> list[dict]:
        """Gibt alle Untertitel-Segmente eines Videos zurück, die innerhalb des definierten Fensters, abhängig von Start oder ende liegen."""
        assert position in {"start", "end"}, "Position muss 'start' oder 'end' sein."
        window_width_ms = int(window_width * 1000)/2
        video_row = self.video_df[self.video_df["id"] == video_id]
        if video_row.empty:
            return []

        subtitles = video_row.iloc[0]["subtitles"]
        sponsors = self.sponsor_df[(self.sponsor_df["videoID"] == video_id) & 
                                   (self.sponsor_df["category"] == "sponsor") &
                                   (self.sponsor_df["votes"] >= 0 )]
        sponsors = self.overlap_filter(sponsors)
        matches = []
        if position == "start":
            for _, s_row in sponsors.iterrows():
                s_start, s_end = seconds_to_ms(s_row["startTime"]) - window_width_ms, seconds_to_ms(s_row["startTime"]) + window_width_ms
                for subtitle in subtitles:
                    sub_start = int(subtitle.get("start_ms", 0))
                    if subtitle.get("duration_ms", None) is None:
                        continue
                    sub_end = sub_start + int(subtitle.get("duration_ms", 0))
                    if sub_end > s_start and sub_start < s_end:
                        matches.append(subtitle)
        elif position == "end":
            for _, s_row in sponsors.iterrows():
                s_start, s_end = seconds_to_ms(s_row["endTime"]) - window_width_ms, seconds_to_ms(s_row["endTime"]) + window_width_ms
                for subtitle in subtitles:
                    sub_start = int(subtitle.get("start_ms", 0))
                    if subtitle.get("duration_ms", None) is None:
                        continue
                    sub_end = sub_start + int(subtitle.get("duration_ms", 0))
                    if sub_end > s_start and sub_start < s_end:
                        matches.append(subtitle)
        
        return matches
    def get_subtitles_within_sponsor_segments_windowed_all(self, position: str, window_width: int = 7, max_videos = None) -> list[dict]:
        """Gibt alle Untertitel-Segmente eines Videos zurück, die innerhalb des definierten Fensters, abhängig von Start oder ende liegen."""
        assert position in {"start", "end"}, "Position muss 'start' oder 'end' sein."
        window_width_ms = int(window_width * 1000)/2

        # Videos filtern
        filtered_videos = self.video_view.filter_by_language()
        #filtered_videos = filtered_videos[filtered_videos["subtitles"].notnull()].copy()
        if max_videos:
            filtered_videos = filtered_videos.head(max_videos)

        # Sponsor-Timestamps vorbereiten
        sponsor_df = self.sponsor_df
        sponsor_df = sponsor_df[
            (sponsor_df["category"] == "sponsor") & 
            ((sponsor_df["votes"] >= 0 ) | (sponsor_df["locked"] == 1)) &
            (sponsor_df["videoID"].isin(self.video_ids_en))
        ]

        # Overlaps filtern
        sponsor_df = sponsor_df.groupby("videoID", group_keys=False).apply(self.overlap_filter)

        sponsor_map = {}
        for _, row in sponsor_df.iterrows():
            sponsor_map.setdefault(row["videoID"], []).append((
                seconds_to_ms(row["startTime"]), 
                seconds_to_ms(row["endTime"])))

        # Sponsor-Text extrahieren
        def extract_text(row):
            assert position in {"start", "end"}
            video_id = row["id"]
            subtitles = row["subtitles"]
            timestamps = sponsor_map.get(video_id, [])
            if not timestamps:
                return ""

            segments = []
            for s_start, s_end in timestamps:
                if position == "start":
                    s_start, s_end = s_start - window_width_ms, s_start + window_width_ms
                elif position == "end":
                    s_start, s_end = s_end - window_width_ms, s_end + window_width_ms
                for seg in subtitles:
                    if seg.get("duration_ms") is None:
                        continue
                    sub_start = seg.get("start_ms", 0)
                    sub_end = sub_start + seg["duration_ms"]
                    if sub_end > s_start and sub_start < s_end:
                        text = seg.get("text", "").replace("\n", " ").replace("\xa0", " ").strip()
                        if text:
                            segments.append(text)
            return " ".join(segments)

        tqdm.pandas(desc="Sponsor-Texte extrahieren")
        filtered_videos["sponsor_text"] = filtered_videos.progress_apply(extract_text, axis=1)
        filtered_videos = filtered_videos[filtered_videos["sponsor_text"].str.strip().astype(bool)]
        
        return filtered_videos

    def extract_n_grams(self, text_collection: list[str], n_gram: int = 3, most_common_k: int = 10, remove_stopwords: bool = False):
        all_ngrams = []
        for text in text_collection:
            # Preprocessing + Tokenisierung
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t.isalpha()] # Keine Zahlen und Sonderzeichen
            # N-Grams generieren
            n_grams = list(ngrams(tokens, n_gram))
            all_ngrams.extend(n_grams)
        # Zählen
        n_gram_count = Counter(all_ngrams)
        return n_gram_count.most_common(most_common_k)
    
    def video_id_iter(self):
        id_list = list(self.video_ids_en)
        for vid_id in id_list:
            yield vid_id
    
    def label_sponsors(self, ner_results_df: pd.DataFrame, min_len: int = 4, max_len: int = 30):
        labeled_sponsors = []

        global_freqs = [normalize_name(ent) for sublist in ner_results_df[ner_results_df["org_entities"].notna()]["org_entities"] for ent in sublist
                        if normalize_name(ent) not in EXCLUDE_ENT and
                        len(normalize_name(ent)) >= min_len and
                        len(normalize_name(ent)) <= max_len]
        global_freqs = Counter(global_freqs)

        # Singleton-Filter
        singleton_ents = set()
        for k, v in global_freqs.items():
            if v == 1:
                singleton_ents.add(k)
        global_freqs = {k:freq for k,freq in global_freqs.items() if k not in singleton_ents}

        total_ents = sum(global_freqs.values())
        global_freqs_rel = {}
        for key, value in global_freqs.items():
            global_freqs_rel[key] = value/total_ents
        
        for _, row in ner_results_df.iterrows():
            video_id = row["video_id"]
            entities = row.get("org_entities", None)
            known_sponsors = row.get("known_sponsor", None)
            if entities is not None and entities.size > 0:
                # Normalisieren, vereinheitlichen Längen-Check und Ausschlussliste
                normed_ent = [normalize_name(e) for e in entities 
                                if normalize_name(e) not in EXCLUDE_ENT and
                                len(normalize_name(e)) >= min_len and 
                                len(normalize_name(e)) <= max_len]
            else:
                normed_ent = []
            if known_sponsors is not None and known_sponsors.size > 0:
                known_sponsors = Counter(known_sponsors)
                normed_ent.extend(known_sponsors)
                labeled_sponsors.append({
                "video_id": video_id,
                "sponsor": normalize_name(known_sponsors.most_common(1)[0][0]),
                "sponsor_ents": normed_ent,
                "raw_entities": entities,
                "known_entities": list(known_sponsors.keys())
                })
                continue
            if not normed_ent:
                continue
            # Ranking anhand lokaler und globaler Frequenz
            local_freq = Counter(normed_ent)
            local_freq_adj = {}
            for key, value in local_freq.items():
                local_freq_adj[key] = value*global_freqs_rel.get(key,0)
            labeled_sponsors.append({
                "video_id": video_id,
                "sponsor": max(local_freq_adj, key=local_freq_adj.get),
                "sponsor_ents": normed_ent,
                "raw_entities": entities,
                "known_entities": None
            })
        return pd.DataFrame(labeled_sponsors)
    
    def get_sponsor_times(self, video_id, filter_overlap: bool = True):
        sponsor_df = self.sponsor_df
        sponsor_df = sponsor_df[
            (sponsor_df["category"] == "sponsor") & 
            ((sponsor_df["votes"] >= 0 ) | (sponsor_df["locked"] == 1)) &
            (sponsor_df["videoID"] == video_id)
        ]
        if sponsor_df.empty:
            return
        if filter_overlap:
            # Overlaps filtern
            sponsor_df = sponsor_df.groupby("videoID", group_keys=False).apply(self.overlap_filter, include_groups = False)
        return sponsor_df
    
    def extract_known_brands(self, text):
        return KNOWN_BRANDS_PATTERN.findall(text)


# Helper

def load_parquets(path, prefix):
    data_dir = Path(path)
    all_parquets = list(data_dir.glob(f"{prefix}*.parquet"))

    df_all = pd.concat([pd.read_parquet(p) for p in all_parquets], ignore_index=True)
    print(f"Datensätze gesamt: {len(df_all)}")
    return df_all

def fast_sponsor_text_worker(args):
    video_id, subtitle_map, sponsor_map = args
    if video_id not in subtitle_map or video_id not in sponsor_map:
        return video_id, ""

    subtitles = subtitle_map[video_id]
    timestamps = sponsor_map[video_id]

    matches = []
    for s_start, s_end in timestamps:
        for seg in subtitles:
            sub_start = seg.get("start_ms", 0)
            if seg.get("duration_ms", None) is None:
                    continue
            sub_end = sub_start + seg.get("duration_ms", 0)
            if sub_end > s_start and sub_start < s_end:
                text = seg.get("text", "").replace("\n", " ").strip()
                if text:
                    matches.append(text)
    return video_id, " ".join(matches)

def seconds_to_ms(seconds) -> int:
    """Konvertiert Sekunden (float) zu Millisekunden (int)."""
    seconds=float(seconds)
    return int(round(seconds * 1000))

def rand_vid_ut_window(dataset, window_size: int = 25, window_pos: str = "start"):
    text=""
    while not text:
        video_id = random.sample(list(dataset.sponsors.video_ids_en), 1)[0]
        text=dataset.sponsors.extract_textfrom_sub(dataset.sponsors.get_subtitles_within_sponsor_segments_windowed(video_id, window_pos, window_size))
    print("VideoID:",video_id)
    return text

def normalize_name(name: str, match: bool = True) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    if match:
        return match_ent(name)
    else:
        return name

def match_ent(name: str) -> str:
    return MATCH_DICT.get(name, name)

def load_dataset():
    df_videos = load_parquets(VIDEO_DATA,"videos_batch_")
    df_channels = load_parquets(CHANNEL_DATA,"channel_data_")
    with open(SPONSOR_DATA, "rb") as f:
        df_sponsors = pickle.load(f)
    ner_results = load_parquets(NER_RESULTS, "merged_ner_250706")
    dataset=YouTubeAdDataset(df_videos, df_channels, df_sponsors, ner_results)
    #dataset=YouTubeAdDataset(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ner_results)
    #dataset=YouTubeAdDataset(df_videos, df_channels, df_sponsors, pd.DataFrame())
    #dataset=YouTubeAdDataset(df_videos, pd.DataFrame(), df_sponsors, ner_results)
    return dataset

if __name__ == "__main__":
    main()