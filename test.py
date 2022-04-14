import spacy
import re
nlp = spacy.load('en_core_web_sm')

sent = "#37-1 Guatemalan Supreme Court approves impeachment of President Molina Yesterday in Guatemala, " \
       "the Supreme Court approved the attorney general's request to impeach President Otto Pérez Molina."
doc = nlp(sent)
for token in doc:
    # print(token, sent.find(str(token), 157, 164))
    if (sent.find(str(token), 157, 164)) == 157:
        print("yeah")

        # Add column for POS tag
        # pos = [(token, token.pos_) for i, sent in enumerate(task_8_data[1])
        #        for token in nlp(sent) if str(token) == str(task_8_data.loc[i, 'strip_tokens'])]

        pos = []
        # for i, sent in enumerate(task_8_data[1][0:20]):
        #     tags = []
        #     for token in nlp(sent):
        #         if str(token) == sent[task_8_data.loc[i, 2]: task_8_data.loc[i, 3]]:
        #             tags.append(token.pos_)
        #             print(token)
        #             print("index find: ", sent.find(str(token), task_8_data.loc[i, 2], (task_8_data.loc[i, 3])))
        #             print("index real: ", task_8_data.loc[i, 2])
