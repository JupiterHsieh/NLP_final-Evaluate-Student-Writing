import streamlit as st
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
import pandas as pd
from transformers import AutoTokenizer,AutoModelForTokenClassification
from collections import defaultdict
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import os
from pathlib import Path
import spacy
from spacy import displacy
from pylab import cm, matplotlib



tokenizer = AutoTokenizer.from_pretrained('checkpoint-2275', add_prefix_space=True,  model_max_length=4096)
model = AutoModelForTokenClassification.from_pretrained('checkpoint-2275', num_labels=15)
data_collator = DataCollatorForTokenClassification(tokenizer,padding='max_length')

trainer = Trainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics, 
)

colors = {
            'Lead': '#8000ff',
            'Position': '#2b7ff6',
            'Evidence': '#2adddd',
            'Claim': '#80ffb4',
            'Concluding Statement': 'd4dd80',
            'Counterclaim': '#ff8042',
            'Rebuttal': '#ff0000',
            'Other': '#007f00',
         }

classes=['Lead',
 'Position',
 'Evidence',
 'Claim',
 'Concluding Statement',
 'Counterclaim',
 'Rebuttal']

def get_class(c):
    if c == 14: return 'Other'
    else: return i2l[c][2:]

def visualize(df, text):
    ents = []

    for i, row in df.iterrows():
        ents.append({
                        'start': int(row['discourse_start']), 
                         'end': int(row['discourse_end']), 
                         'label': row['discourse_type']
                    })

    doc2 = {
        "text": text,
        "ents": ents,
    }

    options = {"colors": colors}
    displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)

def pred2span(pred, example, viz=True, test=False,contents=""):

    n_tokens = len(example['input_ids'])
    classes = []
    all_span = []
    for i, c in enumerate(pred.tolist()):
        if i == n_tokens-1:
            break
        if i == 0:
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
        elif i > 0 and (c == pred[i-1] or (c-7) == pred[i-1]):
            cur_span[1] = example['offset_mapping'][i][1]
        else:
            all_span.append(cur_span)
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
    all_span.append(cur_span)
    
    text = contents
    
    
    # map token ids to word (whitespace) token ids
    predstrings = []
    for span in all_span:
        span_start = span[0]
        span_end = span[1]
        before = text[:span_start]
        token_start = len(before.split())
        if len(before) == 0: token_start = 0
        elif before[-1] != ' ': token_start -= 1
        num_tkns = len(text[span_start:span_end+1].split())
        tkns = [str(x) for x in range(token_start, token_start+num_tkns)]
        predstring = ' '.join(tkns)
        predstrings.append(predstring)
                    
    rows = []
    for c, span, predstring in zip(classes, all_span, predstrings):
        e = {
            'discourse_type': c,
            'predictionstring': predstring,
            'discourse_start': span[0],
            'discourse_end': span[1],
            'discourse': text[span[0]:span[1]+1]
        }
        rows.append(e)


    df = pd.DataFrame(rows)
    df['length'] = df['discourse'].apply(lambda t: len(t.split()))
    
    # short spans are likely to be false positives, we can choose a min number of tokens based on validation
    df = df[df.length > 6].reset_index(drop=True)
    if viz: visualize(df, text)

    return df

def tokenizing(examples):

    o = tokenizer(examples['text'], truncation=True, padding=True, return_offsets_mapping=True, max_length=4096, stride=128, return_overflowing_tokens=True)
    sample_mapping = o["overflow_to_sample_mapping"]
    offset_mapping = o["offset_mapping"]

    return o


tags = defaultdict()
for i, c in enumerate(classes):
    tags[f'B-{c}'] = i
    tags[f'I-{c}'] = i + len(classes)
tags[f'O'] = len(classes) * 2
tags[f'Special'] = -100 
l2i = dict(tags)
i2l = defaultdict()
for k, v in l2i.items(): 
    i2l[v] = k
i2l[-100] = 'Special'
i2l = dict(i2l)
N_LABELS = len(i2l) - 1




st.title("Evaluating Student Writing") 
with st.form(key='myform',clear_on_submit=True):
    contents = st.text_area("Do your writings here!")
    submit_botton = st.form_submit_button("Submit")

    data = [contents,'I','I','I','I','I','I','I']  #一個model 的batch要8個輸入 多得補I
    df = pd.DataFrame(data, columns=['text'])
    dataset = Dataset.from_pandas(df) #把整個batch轉換成huggingface 的dataset

    tokenized_dataset = dataset.map(tokenizing, batched=True,batch_size=8) #map過去
    predictions, labels, _ = trainer.predict(tokenized_dataset) #predict
    preds = np.argmax(predictions, axis=-1)



    
if submit_botton: 
    st.text_area('HIGHLIGHTED FORM')
    with st.expander("Click to read more"):
        st.write("字串結果")