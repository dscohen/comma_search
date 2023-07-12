#create a localhost server to search for the user's input via a search bar and return the results
import string

import pickle
from transformers import AutoTokenizer
import os
import json
import argparse
from pyserini.search import FaissSearcher
import numpy as np
import ast
from sklearn.cluster import KMeans



# load pyserini index
def load_index(index_path):
    #expand
    index_path = os.path.expanduser(index_path)
    searcher = FaissSearcher(index_path, 'castorini/tct_colbert-v2-hnp-msmarco')
    return searcher

def load_text(raw_jsonl='~/data/comma_discord/comma.jsonl'):
    #expand
    raw_jsonl = os.path.expanduser(raw_jsonl)
    with open(raw_jsonl, 'r') as f:
        lines = f.readlines()
    jsonl_dicts = [json.loads(line) for line in lines]
    id_to_text = {jsonl_dict['id']: jsonl_dict['contents'] for jsonl_dict in jsonl_dicts}
    for k,v in id_to_text.items():
        v = v.split('\n')
        id_to_text[k] = {'content': v[0], 'channel': v[1], 'timestamp': v[2], 'author': v[3], 'isPinned': v[4]}

    return id_to_text

def search(searcher, id_to_text, query, k=1000):
    q_emb, hits = searcher.search(query, k, return_vector=True)
    output_ids = []
    embeddings = []
    scores = []
    for hit in hits:
        output_ids.append(hit.docid)
        embeddings.append(hit.vectors)
        scores.append(hit.score)
    embeddings = np.array(embeddings)
    return q_emb, output_ids, scores, embeddings

def clean_output(output,id_to_text):
    out_text = []
    for docid in output:
        #pretty print the output to the user (content, channel, author, ts, pinned)
        id = docid 
        content = id_to_text[id]['content']
        channel = id_to_text[id]['channel']
        author = id_to_text[id]['author']
        ts = id_to_text[id]['timestamp']
        pinned = id_to_text[id]['isPinned']
        out_text.append(f"Content: {content}\nChannel: {channel}\nAuthor: {author}\nTimestamp: {ts}\nPinned: {pinned}\n")
    return '\n'.join(out_text)

def main():
    searcher = load_index('~/data/comma_discord/colbert_encoded')
    id_to_text = load_text()
    while True:
        query = input("Enter query: ")
        # query = 'testing'
        #check if int at end
        qsplit = query.split()
        if qsplit[-1].isdigit():
            k = int(qsplit[-1])
            query = ' '.join(qsplit[:-1])
        else:
            k = 1000

        q_emb, output_ids, scores, embeddings = search(searcher, id_to_text, query, k)
        tokenizer = AutoTokenizer.from_pretrained('castorini/tct_colbert-v2-hnp-msmarco')

        tf_idf_dict = token_idf(tokenizer, id_to_text)
        V, centroids = colbert_prf(searcher, tf_idf_dict, id_to_text, tokenizer, query, 5, embeddings, 24)
        print (f'------Pre PRF: {clean_output(output_ids[:10],id_to_text)}')
        output_ids = prf_with_centroids(output_ids, scores, embeddings, centroids, V, searcher)
        print ('##############################################')
        print (f'------Post PRF:\n {clean_output(output_ids[:10],id_to_text)}')
        output = clean_output(output_ids[:10],id_to_text)

        print(output)
        print (f"Found {len(output_ids)} results ###########################################")
        # break


def token_idf(tokenizer, id_to_text):
    if os.path.exists('idf_dict.pkl'):
        #highest idf terms
        print ("Loading idf dict")

        idf_dict = pickle.load(open('idf_dict.pkl', 'rb'))
        print ("Loaded idf dict")
        highest_idf = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
        print ("Highest idf terms: ")
        for i in range(10):
            print (tokenizer.decode([highest_idf[i][0]]), highest_idf[i][1])
        print ("Lowest idf terms: ")
        for i in range(10):
            print (tokenizer.decode([highest_idf[-i][0]]), highest_idf[-i][1])
        return idf_dict

    idf_dict = {}
    for id in id_to_text:
        text = id_to_text[id]['content']
        token_ids = tokenizer.encode(text)
        token_ids = np.unique(token_ids)
        for tid in token_ids:
            if tid in idf_dict:
                idf_dict[tid] += 1
            else:
                idf_dict[tid] = 1
    #get idf
    for tid in idf_dict:
        idf_dict[tid] = np.log(len(id_to_text)/idf_dict[tid])
    pickle.dump(idf_dict, open('idf_dict.pkl', 'wb'))
    return idf_dict





def get_doc(text_to_id, docid):
    return text_to_id[docid]['content']

def colbert_prf(faiss_searcher, idf_dict, retriever, tokenizer, query, num_fb_docs, embeddings, num_fb_terms, k=10):
    '''
    faiss_searcher: faiss searcher object
    idf_dict: dict of token_id: idf
    retriever: dict of docid: text
    tokenizer: tokenizer object
    query: query string
    num_fb_docs: number of feedback documents to use
    embeddings: embeddings of feedback documents
    num_fb_terms: number of centroids to use in feedback'''
    A = embeddings
    V = []
    decoded_V = []
    #compute kmeans and get centroids
    kmeans = KMeans(n_clusters=num_fb_terms, random_state=0 ).fit(A)
    trans_table = str.maketrans('', '', string.punctuation)
    for c_i, c in enumerate(kmeans.cluster_centers_):
        # search faiss with centroid as query and get top num_fb_docs documents
        results = faiss_searcher.search(c.reshape(1, -1), num_fb_docs)
        # get the doc ids and doc text
        all_cluster_terms_counts = []
        for i in range(num_fb_docs):
            message = get_doc(retriever, results[i].docid)
            #strip punctuation and newlines
            message = message.translate(trans_table)
            
            message = message.replace('\n', ' ')

            #get token_ids with tctcolbert, don't include special tokens
            token_ids = tokenizer.encode(message, add_special_tokens=False)
            #get token counts
            token_counts = np.unique(token_ids, return_counts=True)
            #add to all_cluster_terms_counts
            all_cluster_terms_counts.append(token_counts)
        #merge counts
        merged_counts = {}
        for t in all_cluster_terms_counts:
            for i in range(len(t[0])):
                if t[0][i] in merged_counts:
                    merged_counts[t[0][i]] += t[1][i]
                else:
                    merged_counts[t[0][i]] = t[1][i]
        #get probability by dividing by total number of tokens
        total_tokens = sum(merged_counts.values())
        for k in merged_counts:
            merged_counts[k] = merged_counts[k]/total_tokens
        #sort and add max term to Vv
        sorted_counts = sorted(merged_counts.items(), key=lambda x: x[1], reverse=True)
        candidate_decode = tokenizer.decode([sorted_counts[0][0]])
        # while sorted_counts[0][0] in [term[0] for term in V]:
        #     candidate_decode = tokenizer.decode([sorted_counts[0][0]])
        #     sorted_counts = sorted_counts[1:]
        if len(sorted_counts) == 0:
            print ("No terms found in cluster")
            continue
        max_tid = sorted_counts[0][0]
        #get idf
        idf = idf_dict[max_tid]
        #add to V
        V.append((max_tid, idf, c_i))
        decoded_V.append(tokenizer.decode([max_tid]))
    #print V decoded back to text, ordered by idf
    print("V: ")
    #get only unique
    V = list(set(V))
    #sort
    V = sorted(V, key=lambda x: x[1], reverse=True)
    #take only those greater than 0.8
    V = [v for v in V if v[1] > 1.8][:num_fb_terms]
    for v in V:
        print(tokenizer.decode([v[0]]))
    #retrun V and centroids
    return V, kmeans.cluster_centers_
    


        
def prf_with_centroids(original_run_ids, original_run_scores, original_embeddings, centroids, V, searcher, beta=0.25):
    '''
    original_run_ids: list of docids from original query
    original_run_scores: list of scores from original query
    centroids: list of centroids
    weights: list of weights for each centroid
    searcher: faiss searcher object
    beta: weight of prf component
    '''
    # get scores from original query
    pre_ids = [docid for docid in original_run_ids]
    pre_scores = [score for score in original_run_scores]
    # for each centroid, get the top 1000 documents and weight score of each doc by centroid weight
    pre_doc_results = { docid:score for docid, score in zip(pre_ids, pre_scores)}
    doc_results = {docid:0 for docid in pre_ids}

    for i in range(len(V)):
        weight = V[i][1]
        c_i = V[i][2]
        centroid = centroids[c_i]


        
        # get top 1000 documents
        centroid_embedding = centroid.reshape(1, -1)
        # results = searcher.search(centroid.reshape(1, -1), 1000)
        #comkpute innert product
        results = np.dot(original_embeddings, centroid_embedding.T)
        # get scores
        scores = results.flatten()
        #
        scores = scores * weight
        # add to doc_results
        for j in range(len(results)):
            docid = pre_ids[j]
            if docid in pre_doc_results:
                doc_results[docid] += scores[j]
            else:
                print ("Docid not found in pre_doc_results")
                assert False
        
    #add new prf scores discounted by beta to original scores
    for docid in doc_results:
        pre_doc_results[docid] = beta*(doc_results[docid]) + (1-beta)*pre_doc_results[docid]

    #return ordered docids descending by score
    sorted_doc_score = sorted(doc_results.items(), key=lambda x: x[1], reverse=True)
    sorted_doc = [docid for docid, score in sorted_doc_score]
    return sorted_doc





        
                




        
        #get tokenids for terms
            #tokenize with 


        


if __name__ == '__main__':
    main()



