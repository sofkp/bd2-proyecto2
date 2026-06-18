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
    
if __name__ == "__main__":
    mockup_input = [
        {"doc_id": "doc001", "chunk_id": "p_01", "tf": {"base": 2, "datos": 3}},
        {"doc_id": "doc002", "chunk_id": "p_01", "tf": {"datos": 2, "busqueda": 4}},
        {"doc_id": "doc003", "chunk_id": "p_03", "tf": {"ciencia": 1, "datos": 3}},
    ]
    
    text_cb = CodebookText(top_k=2)
    codebook = text_cb.build_codebook(mockup_input)
    print(codebook)

