import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.DatasetReader import GetAllAminoacids


def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def calculate_distance(aminoacids):
    manhattan_dist = np.zeros((len(aminoacids), len(aminoacids)))
    euclidean_dist = np.zeros((len(aminoacids), len(aminoacids)))
    cosine_dist = np.zeros((len(aminoacids), len(aminoacids)))

    for i in range(len(aminoacids)):
        for j in range(i+1, len(aminoacids)):
            vec1, vec2 = aminoacids[i].embedding, aminoacids[j].embedding

            manhattan_dist[i, j] = manhattan_distance(vec1, vec2)
            euclidean_dist[i, j] = euclidean_distance(vec1, vec2)
            cosine_dist[i, j] = cosine_distance(vec1, vec2)

    return manhattan_dist, euclidean_dist, cosine_dist


def plot_heatmap(words, distances, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, xticklabels=words,
                yticklabels=words, annot=True, cmap='viridis')
    plt.title(title)
    plt.show()
