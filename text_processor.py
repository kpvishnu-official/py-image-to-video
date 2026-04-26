import os
import re


class TextProcessor:
    """
    Reads a plain text file and splits it into chunks.
    Each chunk will become one image + one audio clip in the final video.
    """

    def __init__(self, file_path: str, chunk_size: int = 300):
        """
        Args:
            file_path  : Path to the .txt story file.
            chunk_size : Approximate number of characters per chunk.
                         Smaller = more images/clips. Default is 300.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Story file not found: {file_path}")

        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunks: list[str] = []

    # ------------------------------------------------------------------ #
    #  Public methods                                                       #
    # ------------------------------------------------------------------ #

    def load_and_split(self) -> None:
        """
        Reads the file and splits the text into chunks.
        Tries to split on paragraph boundaries first,
        then falls back to sentence boundaries.
        """
        print(f"  [TextProcessor] Reading file: {self.file_path}")
        raw_text = self._read_file()
        self.chunks = self._split_into_chunks(raw_text)
        print(f"  [TextProcessor] Split into {len(self.chunks)} chunks.")

    def get_chunks(self) -> list[str]:
        """Returns the list of text chunks after load_and_split() is called."""
        if not self.chunks:
            raise RuntimeError("No chunks found. Did you call load_and_split() first?")
        return self.chunks

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _read_file(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Splits text into chunks of roughly `chunk_size` characters.
        Respects paragraph breaks (double newline) and sentence endings.
        """
        # First, try splitting on paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph keeps us under the limit, keep going
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + para
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If a single paragraph is longer than chunk_size, split by sentence
                if len(para) > self.chunk_size:
                    sentence_chunks = self._split_by_sentence(para)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]

    def _split_by_sentence(self, text: str) -> list[str]:
        """Splits a long paragraph into sentence-level chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
        if current:
            chunks.append(current.strip())
        return chunks
