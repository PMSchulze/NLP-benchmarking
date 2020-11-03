import re

# Write function to prepare data for usage 
# with transformers.LineByLineTextDataset.
# That is, we concatenate and use a separate line for the text 
# of each document.
def prepare_linebyline(input_file_path, output_file_path):
    doc = ['']
    with open(input_file_path, encoding="utf-8") as f:
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
    doc.pop()
    with open(output_file_path, 'w') as text_file:
        for line in doc:
            if len(line)>=20:
                print(line, file = text_file)

# Write function to prepare data for usage 
# with transformers.LineByLineTextDataset with block_size=n.
# That is, we concatenate and use a separate line for the text 
# of each document AND jump to new line if line_length>n after the end of the
# last sentence.
def prepare_linebyline_n(input_file, output_file_path, n):
    docs = []
    for line in input_file:
        line_split = re.split("\s+\.|\!|\?", line)
        line_split = [l.strip()+' . ' for l in line_split if l!='\n']
        l = len(line_split)
        i = 0
        truncated_lines = []
        while i<l:
            truncated_line = line_split[i]
            if i == l-1:
                truncated_lines.append(truncated_line)
                break
            for _ in range(i,l-1):
                if len(truncated_line)<n:
                    i += 1
                    truncated_line += line_split[i]        
                else:
                    break
            truncated_lines.append(truncated_line)
            i += 1
        if truncated_lines!=['']:
            docs.extend(truncated_lines)
    docs = list(filter(None,docs))
    with open(output_file_path, 'w') as text_file:
        for line in docs:
            print(line, file = text_file)


# Write function to split a textfile into two part: one which contains the p
# shortest documents, and another one which ontains the remaining 1-p largest
# documents.
def split_documents_by_len(input_file,p):
    doc, docs_short, docs_long = [], [], []
    with open(input_file, encoding='utf-8') as f:
        for i, l in enumerate(f):
            doc.append(l)
        doc.sort(key=len)
        split_line = round((i+1)*p)
        docs_short = doc[:split_line]
        docs_long = doc[split_line:]
    return docs_short, docs_long


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
    del(doc[-1]); del(doc[-1][-1])
    with open(output_file, 'w') as text_file:
        for item in doc:
            for i in item:
                print(i, file = text_file)
