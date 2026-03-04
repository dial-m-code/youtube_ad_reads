"""
Datenexploration Korpus:
Plots für Verteilungen der Standard-Sprache, Kanal-Kategorien und Kanäle mit den meisten Videos.

"""

import re

from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_parquet('/Users/lmeyer/Programmierung DH/Projektarbeit_YT/processed_data/sponsorblock_added_channel_video_metadata_ONLY_SPONSORS.parquet', engine="fastparquet")
    languages = df.drop_duplicates(subset="videoID")
    languages = df["snippet.defaultLanguage"].apply(lambda x: x.split("-")[0] if x is not None else x).dropna()

    # Plot 1 - Verteilung Sprachen
    plt.figure(figsize=(12,6))
    sns.countplot(languages, order = languages.value_counts().index[:10], stat="percent")
    plt.tight_layout()
    plt.show()
    print(languages.value_counts(normalize=True))
  
    df=df[ (df['snippet.defaultAudioLanguage']=="en") |
          (df['snippet.defaultLanguage']=="en")]
    print(df.columns)

    topics = df.groupby("channel_id")["topic_categories"].apply(lambda x: x.dropna())
    channel = df["channel_title_x"].dropna()

    topics_channel = []
    for topic_list in topics:
        for topic in topic_list:
            topics_channel.append(
                re.sub(r"https://en.wikipedia.org/wiki/", "", topic)
                )
    count_t = Counter(topics_channel)
    count_c = Counter(channel)
    print(count_t.most_common(20))
    print(count_c.most_common(20))

    # Plot 2 - Verteilung Kategorien
    series_topics = pd.Series(topics_channel)
    df_cat_count = pd.DataFrame.from_dict(count_t, orient="index", columns=["count"])
    df_cat_count.sort_values("count", inplace=True, ascending=False)
    # sns.countplot(series_topics, order=df_cat_count.index[:10], stat="percent")
    plt.figure(figsize=(12,6))
    sns.countplot(series_topics, order=df_cat_count.index[:10], stat="percent")
    plt.tight_layout()
    plt.show()

    # Plot 3 - Verteilung Kanäle
    df_channel = pd.DataFrame.from_dict(count_c, orient="index", columns=["count"])
    df_channel.sort_values("count", ascending=False, inplace=True)
    df_channel.reset_index(names="channel", inplace=True)
    
    merged = df_channel.merge(df, how="left", left_on="channel", right_on="channel_title_x")
    videos_per_channel = merged.groupby("channel")["videoID"].nunique().reset_index(name="video_count")
    videos_per_channel.sort_values("video_count", inplace=True, ascending=False)
    
    plt.rcParams["font.family"] = "Arial Unicode MS"
    plt.figure(figsize=(12,6))
    sns.barplot(videos_per_channel[:20], x="channel", y="video_count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    df=main()