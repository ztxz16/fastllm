import asyncio
import logging
import json
import traceback
from fastapi import Request

from .protocal.openai_protocol import *

class FastLLmReranker:
  def __init__(self,
               model_name,
               model):
    self.model_name = model_name
    self.model = model

  def rerank(self, request: RerankRequest, raw_request: Request):
      query = request.query
      pairs = []
      for text in request.texts:
         pairs.append([query, text])
      scores = self.model.reranker_compute_score(pairs = pairs)
      ret = []
      for i in range(len(request.texts)):
        now = {'index': i, 'score': scores[i]}
        if (request.return_text):
           now['text'] = request.texts[i]
        ret.append(now)
      ret = sorted(ret, key = lambda x : -x['score'])
      return ret
