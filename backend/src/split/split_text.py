import os
import re

MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 800
DEFAULT_ENCODING = "utf-8"

class SplitText:
    def __init__(self, min_chars=MIN_CHUNK_CHARS, max_chars=MAX_CHUNK_CHARS):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def split_file(self, file_path: str, document_id: str = None):
        text = self.read_txt(file_path)

        if document_id is None:
            document_id = self.get_document_id(file_path)

        return self.split_text(text, document_id, file_path)

    def read_txt(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"no existe el archivo: {file_path}")

        with open(file_path, "r", encoding=DEFAULT_ENCODING) as file:
            return file.read()

    def split_text(self, text: str, document_id: str, source_path: str = None):
        text = self.clean_text(text)
        paragraphs = self.get_paragraphs(text)

        chunks = []
        chunk_index = 0

        for paragraph in paragraphs:
            if len(paragraph) < self.min_chars:
                continue

            parts = self.split_large_paragraph(paragraph)

            for part in parts:
                chunk = {
                    "chunk_id": f"{document_id}_text_{chunk_index}",
                    "document_id": document_id,
                    "modality": "text",
                    "chunk_index": chunk_index,
                    "content": part,
                    "metadata": {
                        "source_path": source_path,
                        "num_chars": len(part),
                    }
                }

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def clean_text(self, text: str):
        if text is None:
            return ""

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def get_paragraphs(self, text: str):
        raw_paragraphs = re.split(r"\n\s*\n", text)

        paragraphs = []
        for paragraph in raw_paragraphs:
            paragraph = paragraph.strip()

            if paragraph != "":
                paragraphs.append(paragraph)

        return paragraphs

    def split_large_paragraph(self, paragraph: str):
        if len(paragraph) <= self.max_chars:
            return [paragraph]

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        parts = []
        curr = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence == "":
                continue

            if len(sentence) > self.max_chars:
                if curr != "":
                    parts.append(curr)
                    curr = ""

                parts.extend(self.split_by_size(sentence))
                continue

            if len(curr) + len(sentence) + 1 <= self.max_chars:
                curr = (curr + " " + sentence).strip()
            else:
                if curr != "":
                    parts.append(curr)
                curr = sentence

        if curr != "":
            parts.append(curr)

        return parts


    def split_by_size(self, text: str):
        parts = []
        start = 0

        while start < len(text):
            end = start + self.max_chars
            part = text[start:end].strip()

            if part != "":
                parts.append(part)

            start = end

        return parts

    def get_document_id(self, file_path: str):
        filename = os.path.basename(file_path)
        filename_without_extension = os.path.splitext(filename)[0]
        return filename_without_extension
    

def print_chunks(chunks):
    print(f"Total chunks generados: {len(chunks)}")

    for chunk in chunks:
        print("\n--------------------------------------------")
        print("chunk_id:", chunk["chunk_id"])
        print("document_id:", chunk["document_id"])
        print("modality:", chunk["modality"])
        print("chunk_index:", chunk["chunk_index"])
        print("num_chars:", chunk["metadata"]["num_chars"])
        print("content:")
        print(chunk["content"][:300])

def test_file():
    file_path = "backend/src/split/test.txt"

    splitter = SplitText()
    chunks = splitter.split_file(file_path)

    print_chunks(chunks)


if __name__ == "__main__":
    test_file()