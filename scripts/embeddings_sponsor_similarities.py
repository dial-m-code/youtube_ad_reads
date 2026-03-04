"""
Konvertiert AD-Reads in TFIDF-Vektoren und vergleicht Ähnlichkeiten zum Sponsor-Zentrum
Druckt Statistiken
Plot zur Visualisierung
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sys import exit

from scipy.stats import bootstrap

N_COMMON_SPON = 100 # only use the most popular sponsors, regardless of how many channels feature them
N_COMMON_CHANNELS = 100
N_MIN_CHANNELS = 2 # minimum amount of channels to feature a sponsor (avoid niche sponsors)
MIN_LEN = 150

def main():
    df_sponsors = pd.read_parquet("/Users/lmeyer/Programmierung DH/Projektarbeit_YT/scripts/sponsors_and_text.parquet")
    df_channels = pd.read_parquet("/Users/lmeyer/Programmierung DH/Projektarbeit_YT/converted_dataset/channel_data_0000.parquet")

    sponsor_stop = set(df_sponsors["sponsor"])
    #stop_list_custom = set(sponsor_stop | ENGLISH_STOP_WORDS)
    stop_list_custom = list(sponsor_stop)


    most_common_sponsors = Counter(df_sponsors["sponsor"]).most_common(N_COMMON_SPON)
    most_common_sponsors_set = set(s for s, _ in most_common_sponsors)

    if N_MIN_CHANNELS is not None:
        df_relevant_sponsors = (df_sponsors
                    .groupby("sponsor")["channel_title"]
                    .nunique()
                    .reset_index(name="no_channels"))
        df_relevant_sponsors = df_relevant_sponsors[df_relevant_sponsors["no_channels"] >= N_MIN_CHANNELS]
        df_relevant_sponsors.reset_index(drop=True, inplace=True)
        df_sponsors_filtered = df_sponsors[df_sponsors["sponsor"].isin(df_relevant_sponsors["sponsor"])]
    else:
        df_sponsors_filtered = df_sponsors[df_sponsors["sponsor"].isin(most_common_sponsors_set)]
    
    df_sponsors_filtered = df_sponsors_filtered[df_sponsors_filtered["sponsor_text"].str.len() > MIN_LEN] 
    df_sponsors_filtered.reset_index(drop=True, inplace=True)
   
    documents = df_sponsors_filtered["sponsor_text"]

    vecorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b',
        min_df = 2,
        max_df= 0.9,
        ngram_range=(1,3),
        stop_words=stop_list_custom
    )
    
    vectorizer_fit = vecorizer.fit_transform(documents)

    all_sponsors = set(df_sponsors_filtered["sponsor"])

    sponsor_means = {}
    for sponsor in all_sponsors:
        ids = df_sponsors_filtered[df_sponsors_filtered["sponsor"] == sponsor].index
        sponsor_vectors = vectorizer_fit[ids]
        mean_vector_sponsor = np.mean(sponsor_vectors, axis=0)
        sponsor_means[sponsor] = mean_vector_sponsor

    channels_most_common = Counter(df_sponsors_filtered["channel_title"]).most_common(N_COMMON_CHANNELS)
    channels_most_common_set = set(c for c, _ in channels_most_common)

    ad_read_distances = {}
    ad_read_texts = []
    for creator in channels_most_common_set:
        ad_reads = df_sponsors_filtered[df_sponsors_filtered["channel_title"] == creator]
        for ad_read_id in ad_reads.index:
            current_sponsor = ad_reads.loc[ad_read_id, "sponsor"]

            ad_read_distance = cosine_similarity(vectorizer_fit[ad_read_id], np.asarray(sponsor_means[current_sponsor]))[0][0]

            ad_read_texts.append({
                "sponsor": current_sponsor,
                "similarity": ad_read_distance,
                "channel": ad_reads.loc[ad_read_id, "channel_title"],
                "channel_subscriber": df_channels[df_channels["id"] == ad_reads.loc[ad_read_id, "snippet"]["channelId"]]["channel_statistics"].iloc[0].get("subscriberCount", -1),
                "video": "https://youtube.com/watch?v=" + ad_reads.loc[ad_read_id, "video_id"],
                "ad_read_text": ad_reads.loc[ad_read_id, "sponsor_text"]
            })

            if creator in ad_read_distances:
                ad_read_distances[creator].append(ad_read_distance)
            else:
                ad_read_distances[creator] = [ad_read_distance]
    
    ad_read_texts = pd.DataFrame(ad_read_texts)
    ad_read_texts.to_csv("ad_read_texts.csv", index=False)
    
    df_cos_sim = dict_list_to_df(ad_read_distances, "channel", "cos_sim")

    channel_means_std = []
    # channel_name, mean, std
    for channel, similarities in ad_read_distances.items():
        channel_means_std.append({
            "channel_name": channel,
            "mean": np.mean(similarities),
            "std": np.std(similarities),
            "subscriber_count": df_channels[df_channels["channel_title"] == channel]["channel_statistics"].iloc[0].get("subscriberCount", -1)
        })
    channel_means_std = pd.DataFrame(channel_means_std)

    channel_means_std.to_csv("means_and_std.csv", index=False)

    # Histgramm
    df_cos_sim["cos_sim"].hist(figsize=(10,4), bins=20)
    plt.xlabel('Cosinus-Ähnlichkeit')
    plt.ylabel('Frequenz')
    plt.tight_layout()
    #plt.savefig("sem_embed_hist_cos_sim.png", dpi = 300)
    plt.show()

    # Bootstrap Mean
    boot_data=(df_cos_sim["cos_sim"].to_numpy(),)
    #print(boot_data)
    res=bootstrap(boot_data, np.mean, n_resamples=10_000, method="percentile")
    print(res)

    # Stats deskriptiv
    print("Deskriptiv:")
    print(f"Mean: {df_cos_sim['cos_sim'].mean():.2f}")
    print(f"Median: {df_cos_sim['cos_sim'].median():.2f}")
    print(f"SD: {df_cos_sim['cos_sim'].std():.2f}")

    print(df_cos_sim.shape[0])

    sns.histplot(res.bootstrap_distribution)
    plt.show()

    # Bootstrap Varianz
    print()
    print("Bootstrap Varianz")
    means_channel = df_cos_sim.groupby("channel")["cos_sim"].mean()
    std_observed = means_channel.std()
    print(f"Beob. STD: {std_observed:.4f}")

    n_resamples = 10_000
    boot_std = []

    channel = df_cos_sim["channel"].unique()

    for i in range(n_resamples):
        boot_mean = []
        for chan in channel:
            channel_data = df_cos_sim.loc[df_cos_sim["channel"] == chan, "cos_sim"].values
            resampling = np.random.choice(channel_data, len(channel_data), replace=True)
            boot_mean.append(np.mean(resampling))
        boot_std.append(np.std(boot_mean))
        if i % 1000:
            print(f"{i} / {n_resamples}")
    
    boot_std = np.array(boot_std)

    confidence_level = 95
    ci_low = np.percentile(boot_std, (100 - confidence_level) / 2)
    ci_high = np.percentile(boot_std, (100 - confidence_level) / 2 + confidence_level)

    print(f"{confidence_level}% Konfidenz-Intervall: [{ci_low:.4f}, {ci_high:.4f}]")

    #exit()

    # Maskieren für Plot
    #mask_ch = list(np.arange(5)) + list(np.arange(N_COMMON_CHANNELS//2+1,N_COMMON_CHANNELS//2 + 6)) + list(np.arange(N_COMMON_CHANNELS-6, N_COMMON_CHANNELS-1))
    mask_ch = None

    if mask_ch:
        order = (
        df_cos_sim
        .groupby("channel")["cos_sim"]
        .mean()
        .sort_values(ascending=False)
        .iloc[mask_ch]
        )
    
        rename_dict = {}
        for item, rank in zip(order.index, mask_ch):
            #item.rename(index=f"({rank+1}) {item.index}", inplace = True)
            rename_dict[item] = f"({rank+1}) {item}"
    
        order.rename(rename_dict, inplace=True)
        df_cos_sim["channel"] = df_cos_sim["channel"].replace(rename_dict)

        order = order.index
    else:
        order = (
        df_cos_sim
        .groupby("channel")["cos_sim"]
        .mean()
        .sort_values(ascending=False)
        .index
        )

    # Plot
    plt.figure(figsize=(12, 6), dpi=300)
    if mask_ch:
        sns.barplot(df_cos_sim[df_cos_sim["channel"].isin(order)], x="channel", y="cos_sim", errorbar="sd", order=order)
    else:
        sns.barplot(df_cos_sim, x="channel", y="cos_sim", errorbar="sd", order=order)

    plt.xlabel('Kanal')
    plt.ylabel('Cosinus-Ähnlichkeit')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("sem_embed_fig_all_2_ch.png")
    plt.show()

def dict_list_to_df(dict_list, name_keys, name_values):
    df_ready_list = []
    for key, l in dict_list.items():
        for item in l:
            df_ready_list.append({
                name_keys: key,
                name_values: item
            })
    return pd.DataFrame(df_ready_list)

if __name__ == "__main__":
    main()