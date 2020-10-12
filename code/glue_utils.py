import os.path
import pandas as pd

def load_data(task, glue_dir):
  header = None if task == 'CoLA' else 0
  df_train = pd.read_csv(os.path.join(glue_dir, task, 'train.tsv'), sep='\t|\\\\t', header = header, engine='python')
  df_eval = pd.read_csv(os.path.join(glue_dir, task, 'dev.tsv'), sep='\t|\\\\t', header = header, engine='python')
  return df_train, df_eval;
  
def extract_cols_single(task, df):
  if task == 'CoLA':
    labels, sentences = df.iloc[:,1].tolist(), df.iloc[:,3].tolist()
  elif task == 'SST-2':
    labels, sentences = df.iloc[:,1].tolist(), df.iloc[:,0].tolist()
  sentences = ["<|startoftext|>"+ x + "<|endoftext|>" for x in sentences]
  return labels, sentences;
      
def extract_cols_NLI(task, df):
  if task == 'QNLI' or task =='RTE' or task =='WNLI':
    labels, premises, hypotheses = df.iloc[:,3].tolist(), df.iloc[:,1].tolist(), df.iloc[:,2].tolist()
  elif task == 'MNLI':
    labels, premises, hypotheses = df.iloc[:,11].tolist(), df.iloc[:,8].tolist(), df.iloc[:,9].tolist()
  sentences = ["<|startoftext|>"+ x + "<$>" + y + "<|endoftext|>" for x,y in zip(premises, hypotheses)]
  return labels, sentences

def extract_and_prepare(task, df):
  if task == 'CoLA' or task == 'SST-2':
    labels, sentences = extract_cols_single(task, df)
  elif task == 'QNLI' or task =='RTE' or task =='WNLI' or task == 'MNLI':
    labels, sentences = extract_cols_NLI(task, df)
  return labels, sentences
