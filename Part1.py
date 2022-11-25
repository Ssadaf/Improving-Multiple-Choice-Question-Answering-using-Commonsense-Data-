#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pickle
import networkx as nx
import spacy
from string import punctuation
import pandas as pd
import numpy as np
import graphviz


# # Reading Data & Preprocessing

# ### Filtering English Words

# In[2]:


english_dataset = []
with open("assertions.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if line[2].startswith('/c/en') and line[3].startswith('/c/en'):
            english_dataset.append(line)


# In[6]:


outfile = open('english_dataset','wb')
pickle.dump(english_dataset,outfile)
outfile.close()


# ### Cleaning Data

# In[4]:


data = []
for line in english_dataset:
    source = line[2].split('/', maxsplit=4)[3]
    target = line[3].split('/', maxsplit=4)[3]
    relation = line[1].split('/', maxsplit=4)[2]
    w = float(line[4].split(", \"weight\": ")[1].split("}")[0])
    data.append((source, target, {"relation" : relation, "weight" : w}))


# In[7]:


outfile = open('data','wb')
pickle.dump(data,outfile)
outfile.close()


# In[26]:


data[10:20]


# # Constructing Graph

# In[28]:


G = nx.DiGraph()
G.add_edges_from(data)


# In[29]:


G.number_of_edges()


# ### Adding Reverse Edges

# In[42]:


reverse_edges = []

for i, e in enumerate(G.edges()):
    if i % 1000 == 0:
        print(i)
        
    source,target = e
    if not G.has_edge(target, source):
        relation = G.edges[source,target]['relation'] + ' -1'
        w = G.edges[source,target]['weight']
        r_edge = (target, source, {"relation" : relation, "weight" : w})
        reverse_edges.append(r_edge)


# In[47]:


len(reverse_edges)
reverse_edges[15]


# In[311]:


outfile = open('reverse_edges','wb')
pickle.dump(reverse_edges,outfile)
outfile.close()


# In[48]:


G.add_edges_from(reverse_edges)
G.number_of_edges()


# In[156]:


outfile = open('graph','wb')
pickle.dump(G,outfile)
outfile.close()


# # Finding Paths

# ### Choosing the Path based on Edge Scores

# In[80]:


def find_max_score_path(G, source, target, cutoff):
    paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=cutoff))
    scores = []
    for path in paths:
        pathGraph = nx.path_graph(path)
        pathScore = 1
        for e in pathGraph.edges():
            pathScore *= G.edges[e[0], e[1]]['weight']
        scores.append(pathScore)
    return paths[np.argmax(scores)]


# In[81]:


def find_path(source_word, target_word):
    shortest_length = nx.shortest_path_length(G, source=source_word, target=target_word)
    
    return find_max_score_path(G, source_word, target_word, shortest_length)


# # Visualizing path

# In[200]:


def visualize_path(G, path):
    edge_types = []
    
    pathGraph = nx.path_graph(path)
    for e in pathGraph.edges():
        print(e, G.edges[e[0], e[1]])

# #     Networkx Visualization
#     subgraph = G.subgraph(path)
#     pos = nx.nx_pydot.graphviz_layout(subgraph)
#     nx.draw_networkx(subgraph, pos, arrows=True)
#     edge_labels = nx.get_edge_attributes(subgraph,'relation')
#     nx.draw_networkx_edge_labels(subgraph, pos, edge_labels = edge_labels)
    
    vg = nx.DiGraph()
    for e in pathGraph.edges():
        rel = G.edges[e[0], e[1]]['relation']
        edge_types.append(rel)
        vg.add_edge(e[0], e[1], label = rel)

    A = nx.nx_agraph.to_agraph(vg)
    A.layout()
    A.draw(path[0] + '_to_' +  path[-1] + '_graph.png')
    display(A)
    
    return len(path)-1, edge_types


# In[201]:


s_path = find_path('flu', 'party')
visualize_path(G, s_path)


# In[202]:


s_path = find_path('girl', 'sleep')
visualize_path(G, s_path)


# ### _________________ From this point it is Step 2

# # Extracting Terms

# In[131]:


nlp = spacy.load("en_core_web_md")


# In[132]:


def extract_keywords(text):
    result = []
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        result.append(token.text)
    return result


# In[258]:


def delete_notexisted_words(words):
    result = []
    for word in words:
        if word in G:
            result.append(word)
    return result


# In[259]:


def find_question_context_keywords(question, context):
    q = extract_keywords(question)
    c = extract_keywords(context)
    keywords = c + q
    keywords = delete_notexisted_words(keywords)
    return keywords


# In[263]:


def find_choices_keywords(choices):
    keywords = []
    for choice in choices:
        keywords.append(delete_notexisted_words(extract_keywords(choice)))
    return keywords


# # Examples

# In[226]:


choice_index = ['(a)', '(b)', '(c)', '(d)', '(e)']


# In[301]:


def find_QandA_paths(G, q, c):
    stats = {}

    for i, choice in enumerate(c):
        print("_____________\n_____________\nChoice:", choice_index[i], "\n")
        min_len_choice = np.inf
        max_len_choice = 0 
        edge_types_choice = []

        for q_word in q:
            print("===\nWord:",q_word, "\n")
            for word in choice:
                if word == q_word:
                    length, edges = 0, []
                    visualize_path(G, [word])
                else:
                    s_path = find_path(q_word, word)
                    length, edges = visualize_path(G, s_path) 

                if (length < min_len_choice) : 
                    min_len_choice = length
                if (length > max_len_choice) : 
                    max_len_choice = length
                edge_types_choice = edge_types_choice + edges
                
        stats[i] = {'min': min_len_choice, 'max': max_len_choice, 'edges': edge_types_choice}
    return stats


# ### Example 1 (CommonSenseQA)
# 
# Question: The only baggage the woman checked was a drawstring bag, where was she heading with it?
# 
# Choices:
# (a) garbage can
# (b) military
# (c) jewelry store
# (d) safe
# (e) airport
# 
# Correct Answer: e

# In[264]:


question = "The only baggage the woman checked was a drawstring bag, where was she heading with it?"
choices = ["garbage can", "military", "jewelry store", "safe", "airport"]


# In[265]:


q_words = find_question_context_keywords(question, "")
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[237]:


ex1 = find_QandA_paths(G, q_words, c_words)


# In[238]:


ex1


# ### Example 2 (COPA)
# 
# Context: The host cancelled the party.
# Question: What was the cause?
# 
# Choices:
# (a) She was certain she had the flu.
# (b) She worried she would catch the flu.
# 
# Correct Answer: a

# In[231]:


context = "The host cancelled the party."
question = "What was the cause?"
choices = ["She was certain she had the flu.","She worried she would catch the flu."]


# In[232]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[233]:


ex2 = find_QandA_paths(G, q_words, c_words)


# In[234]:


ex2


# ### Example 3 (COPA)
# 
# Context: The man uncovered incriminating evidence against his enemy.
# Question: What happened as a result?
# 
# Choices:
# (a) The man avoided his enemy.
# (b) The man blackmailed his enemy.
# 
# Correct Answer: b

# In[245]:


context = "The man uncovered incriminating evidence against his enemy."
question = "What happened as a result?"
choices = ["The man avoided his enemy.","The man blackmailed his enemy."]


# In[246]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[247]:


ex3 = find_QandA_paths(G, q_words, c_words)


# In[248]:


ex3


# ### Example 4 (Social IQa)
# 
# Context: Alex had a party at his house while his parents were out of town even though they told him not to.
# Question: What will happen to Alex's parents?
# 
# Choices:
# (a) punish Alex for having a party.
# (b) get in trouble.
# (c) have to clean up the mess.
# 
# Correct Answer: a

# In[249]:


context = "Alex had a party at his house while his parents were out of town even though they told him not to."
question = "What will happen to Alex's parents?"
choices = ["punish Alex for having a party.","get in trouble.", "have to clean up the mess."]


# In[250]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[251]:


ex4 = find_QandA_paths(G, q_words, c_words)


# In[252]:


ex4


# ### Example 5 (Social IQa)
# 
# Context: Skylar got a letter in the mail, it was from harvard, he was excited.
# Question: What will Skylar want to do next?
# 
# (a) read the letter.
# (b) apply for entry into Harvard. 
# (c) throw the letter away.
# 
# Correct Answer: a

# In[266]:


context = "Skylar got a letter in the mail, it was from harvard, he was excited."
question = "What will Skylar want to do next?"
choices = ["read the letter.","apply for entry into Harvard. ", "throw the letter away."]


# In[267]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[269]:


ex5 = find_QandA_paths(G, q_words, c_words)


# In[270]:


ex5


# ### Example 6 (WinoGrande)
# 
# Context: They eased the pipe onto the giant pile so it wouldn't burst as it was rather rigid.	
# Question: What was rather rigid?
# 
# Choices:
# (a) pipe
# (b) pile
# 
# Correct Answer: a

# In[271]:


context = "They eased the pipe onto the giant pile so it wouldn't burst as it was rather rigid."
question = "What was rather rigid?"
choices = ["pipe","pile"]


# In[272]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[273]:


ex6 = find_QandA_paths(G, q_words, c_words)


# In[274]:


ex6


# ### Example 7 (WinoGrande)
# 
# Context: Visiting New York City interested Kevin but not Lawrence because he hates being around crowds of people.
# Question: What hates being around crowds of people?
# 
# Choices:
# (a) Kevin
# (b) Lawrence
# 
# Correct Answer: b

# In[275]:


context = "Visiting New York City interested Kevin but not Lawrence because he hates being around crowds of people."
question = "What hates being around crowds of people?"
choices = ["Kevin","Lawrence"]


# In[276]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[277]:


ex7 = find_QandA_paths(G, q_words, c_words)


# In[278]:


ex7


# ### Example 8 (RocStories)
# 
# Context: Jim got his first credit card in college. He didn’t have a job so he bought everything on his card. After he graduated he amounted a $10,000 debt. Jim realized that he was foolish to spend so much money.
# Question: What is the next sentence?
# 
# Choices:
# (a) Jim decided to open another credit card.
# (b) Jim decided to devise a plan for repayment.
# 
# Correct Answer: b

# In[279]:


context = "Jim got his first credit card in college. He didn’t have a job so he bought everything on his card. After he graduated he amounted a $10,000 debt. Jim realized that he was foolish to spend so much money."
question = "What is the next sentence?"
choices = ["Jim decided to open another credit card.","Jim decided to devise a plan for repayment."]


# In[280]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[281]:


ex8 = find_QandA_paths(G, q_words, c_words)


# In[282]:


ex8


# ### Example 9 (MCTaco)
# 
# Context: Growing up on a farm near St. Paul, L. Mark Bailey didn't dream of becoming a judge.
# Question: What did Mark do right after he found out that he became a judge?
# 
# Choices:
# (a) had a nice dinner
# (b) he buy a 45-acre horse farm
# (c) he celebrated
# (d) mark retired
# (e) he dream of participating himself
# 
# Correct Answer: a, c

# In[283]:


context = "Growing up on a farm near St. Paul, L. Mark Bailey didn't dream of becoming a judge."
question = "What did Mark do right after he found out that he became a judge?"
choices = ["had a nice dinner","he buy a 45-acre horse farm", "he celebrated", "mark retired", "he dream of participating himself"]


# In[284]:


q_words = find_question_context_keywords(question, context)
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[285]:


ex9 = find_QandA_paths(G, q_words, c_words)


# In[286]:


ex9


# ### Example 10 (PIQA)
# 
# Question: What should you do to separate egg whites from the yolk using a water bottle?
# 
# Choices:
# (a) Squeeze the water bottle and press it against the yolk. Release, which creates suction and lifts the yolk.
# (b) Place the water bottle and press it against the yolk. Keep pushing, which creates suction and lifts the yolk.
# 
# Correct Answer: a

# In[302]:


question = "What should you do to separate egg whites from the yolk using a water bottle?"
choices = ["Squeeze the water bottle and press it against the yolk. Release, which creates suction and lifts the yolk.",
           "Place the water bottle and press it against the yolk. Keep pushing, which creates suction and lifts the yolk."]


# In[303]:


q_words = find_question_context_keywords(question, "")
c_words = find_choices_keywords(choices)
print("Question words:", q_words)
print("Choices words:", c_words)


# In[304]:


ex10 = find_QandA_paths(G, q_words, c_words)


# In[305]:


ex10


# # Statistics of Knoweledge Types

# In[329]:


experiments = [ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10]


# In[330]:


exp_edges = []

for experiment in experiments:
    for key in experiment:
        data = experiment[key]
        exp_edges += data['edges']


# In[335]:


import collections
x = collections.Counter(exp_edges)
dict(sorted(x.items(), key=lambda item: item[1]))


# In[ ]:




