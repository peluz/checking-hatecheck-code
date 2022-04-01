import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


def preprocess_texts(df, text_col, label_col, min_df=1):
    vectorizer = CountVectorizer(min_df=min_df)
    labelEncoder = LabelEncoder()
    tokenized_texts = vectorizer.fit_transform(df[text_col])
    labels = labelEncoder.fit_transform(df[label_col])
    return vectorizer, labelEncoder, tokenized_texts, labels

class HateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def get_results(trainer, dataset):
    results = trainer.predict(dataset)
    for metric in results.metrics:
        print(metric, results.metrics['{}'.format(metric)])
    preds=[]
    for row in results[0]:
        preds.append(int(np.argmax(row)))
    print(classification_report(dataset.labels,preds,digits=4))
    return results, preds