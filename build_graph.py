import re
from collections import defaultdict


def build_attr_graph(items, mapping, similarity_matrix=None, tau=0.1):
    edges = []
    text_hashtag_map = defaultdict(set)
    text_users_map = defaultdict(set)
    img_hashtag_map = defaultdict(set)
    img_users_map = defaultdict(set)
    if similarity_matrix is None:
        raise 'similarity_matrix is None'

    for it in items:
        tweet = it['tweet_text']
        hashtags = re.findall(r"#\w+", tweet)
        mentioned_users = re.findall(r"@\w+", tweet)
        hts = [ht[1:].lower() for ht in hashtags]
        users = [user[1:].lower() for user in mentioned_users]

        src_idx, trg_idx = mapping[it['tweet_id']], mapping[it['image_id']]

        for ht in hts:
            text_hashtag_map[ht].add(src_idx)
            img_hashtag_map[ht].add(trg_idx)
        for u in users:
            text_users_map[u].add(src_idx)
            img_users_map[u].add(trg_idx)

    for tweet_ids in text_hashtag_map.values():
        tweet_ids = list(tweet_ids)
        for i in range(len(tweet_ids)):
            for j in range(i + 1, len(tweet_ids)):
                edges.append([i, j])

    for tweet_ids in img_hashtag_map.values():
        tweet_ids = list(tweet_ids)
        for i in range(len(tweet_ids)):
            for j in range(i + 1, len(tweet_ids)):
                edges.append([i, j])


    for tweet_ids in text_users_map.values():
        tweet_ids = list(tweet_ids)
        for i in range(len(tweet_ids)):
            for j in range(i + 1, len(tweet_ids)):
                edges.append([i, j])
                # if similarity_matrix[i, j] >= tau:
                #     edges.append([i, j])


    for tweet_ids in img_users_map.values():
        tweet_ids = list(tweet_ids)
        for i in range(len(tweet_ids)):
            for j in range(i + 1, len(tweet_ids)):
                edges.append([i, j])

    # flattened = [n for e in edges for n in e]
    # print(f"\tnum of Nodes: ", len(set(flattened)))
    # print('\tnum of attr edges', len(edges))

    return edges

if __name__ == '__main__':
    edges = build_attr_graph()