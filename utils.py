import streamlit as st
import requests
from typing import List
import io
import shutil
import pytesseract  
import base64
import requests
import re
import json
import time
from datetime import datetime
from scipy.stats import norm
import numpy as np
from typing import Tuple
from FlagEmbedding import FlagReranker
import os
import openai
import ffmpeg
import mammoth
import tempfile
# from win32com import client
import whisper
from sentence_transformers import CrossEncoder
'''
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
'''