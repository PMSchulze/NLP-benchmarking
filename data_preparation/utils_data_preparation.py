import re

# Write function to prepare data for usage 
# with transformers.LineByLineTextDataset.
# That is, we concatenate and use a separate line for the text 
# of each document.
def prepare_linebyline(input_file, output_file):
    doc = ['']
    with open(input_file, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if len(line) != 0:
                if line[0] != '=':
                    doc[-1] += line
            else:
                if doc[-1] != '':
                    doc.append('')
    with open(output_file, 'w') as text_file:
        for i, line in enumerate(doc):
            if doc[i+1] is not None:
                print(line, file = text_file)

# Write function to prepare data for usage 
# with transformers.TextDatasetForNextSentencePrediction.
# That is, we place each sentence on a separate line and add blank lines 
# between documents.
def prepare_nextsentence(input_file, output_file):
    doc = ['']
    with open(input_file, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = re.split("\s+\.|\!|\?", line)
            doc[-1] = [l.strip()+' .' if l!='\n' else '' for l in line]
            doc.append('')
    with open(output_file, 'w') as text_file:
        for item in doc:
            for i in item:
                print(i, file = text_file)
