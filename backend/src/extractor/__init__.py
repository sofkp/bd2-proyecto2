# Exporta las clases públicas del módulo extractor
from .base import BaseExtractor
from .tfidf import TFIDFExtractor
from .sift import SIFTExtractor
from .mfcc import MFCCExtractor

__all__ = ["BaseExtractor", "TFIDFExtractor", "SIFTExtractor", "MFCCExtractor"]
