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
def prepare_linebyline_n(input_file, n):
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
    return docs


# Write function to split a textfile into two part: one which contains the p
# shortest documents, and another one which ontains the remaining 1-p largest
# documents.
def split_documents_by_len(input_file_path,p):
    doc, docs_short, docs_long = [], [], []
    with open(input_file_path, encoding='utf-8') as f:
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


# Take two textfiles as input, one with short documents on each line, and
# another one with long documents on each line. Then, split each document into
# smaller chunks (of length len_short and len_long, respectively), drop chunks
# with length<20 characters, and finally transfer all short chunks to short
# file (these are leftovers from the long cunks).  
def divide_into_chunks(
    input_file_short, input_file_long, len_short, len_long
):
    docs_short = prepare_linebyline_n(
        input_file = input_file_short,
        n = len_short
    )
    docs_long = prepare_linebyline_n(
        input_file = input_file_long,
        n = len_long
    )
    docs_short_tmp, docs_long_tmp = [], []
    docs_short_tmp = [doc for doc in docs_short if len(doc)>=20]
    docs_long_tmp = [doc for doc in docs_long if len(doc)>=20]
    docs_short_out, docs_long_out = docs_short_tmp, []
    for doc in docs_long_tmp:
        if len(doc)<len_short:
            docs_short_out.append(doc)
        else:
            docs_long_out.append(doc)
    return docs_short_out, docs_long_out

