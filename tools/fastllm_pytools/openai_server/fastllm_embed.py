import asyncio
import logging
import json
import traceback
from fastapi import Request

from .protocal.openai_protocol import *

class FastLLmEmbed:
  def __init__(self,
               model_name,
               model):
    self.model_name = model_name
    self.model = model

  def embedding_sentence(self, request: EmbedRequest, raw_request: Request):
      return self.model.embedding_sentence(request.inputs, request.normalize)
