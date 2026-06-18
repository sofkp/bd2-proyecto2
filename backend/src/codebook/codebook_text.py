import math
from collections import defaultdict

class CodebookText:
    def __init__(self, top_k):
        self.top_k= top_k
        self.codebook= {}

    def build_codebook(self, chunks):
        chunk_freq= defaultdict(int)
        total_chunks= len(chunks)

        for chunk in chunks:
            unique_words= set(chunk['tf'].keys())
            for word in unique_words:
                chunk_freq[word] += 1
        
        sorted_by_freq= sorted(chunk_freq.items(), key=lambda item: item[1], reverse=True)
        top_k_terms= sorted_by_freq[:self.top_k]
        for index, (word, df) in enumerate(top_k_terms):
            idf= math.log(total_chunks / df) + 1
            self.codebook[word]= {"index": index, "idf": round(idf, 4)}
    
        return self.codebook
