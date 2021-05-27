#!/usr/bin/python3
import os
import sys
import signal
import argparse
from concurrent import futures
from itertools import product
import warnings
import time
import struct
import numpy as np
import grpc
from timeit import default_timer as now

import torch
import seq2seq.utils as utils
import seq2seq.data.config as config
from seq2seq.data.dataset import RawTextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.translator import Translator
from seq2seq.models.gnmt import GNMT
from seq2seq.inference import tables
from seq2seq.inference.beam_search import SequenceGenerator

sys.path.insert(0, '../')
import infaas_request_status_pb2 as infaas_status
import query_pb2 as infaas_query
import query_pb2_grpc as infaas_query_grpc

model_directory = '/tmp/model'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_GRPC_MSG_SIZE = 2147483647
MAX_SEQ_LEN = 200  # Maximum generated sequence length
model_suffix = 'pth'

def terminateHandler(sigNum, frame):
  print('Received: {}, exit process.'.format(sigNum))
  sys.exit()

def get_args():
  ap = argparse.ArgumentParser()
  ap.add_argument('--cuda', dest='cuda', action='store_true',
                  help='Use cuda (GPU) for inference.')
  ap.add_argument('--no-cuda', dest='cuda', action='store_false',
                  help='Use CPU for inference.')
  ap.set_defaults(cuda=False)
  ap.add_argument('--math', dest='math', default='fp32',
                  choices=['fp16', 'fp32'],
                  help='Precision. FP16 only supported on GPU.')

  ap.add_argument('--beam-size', dest='beam_size', type=int, default=2,
                  choices=[1, 2, 5],
                  help='Beam size (e.g., 1, 2, 5)')
  ap.add_argument('--model', dest='model', default='testmodel',
                  help='Model name to load')
  ap.add_argument('--port', dest='port', type=int, default=9001,
                  help='Port to listen on')

  args = ap.parse_args()
  # Validate the input: CPU can only use FP32.
  if 'fp16' in args.math and not args.cuda:
    ap.error('--math fp16 requires --cuda True')

  return args

class ModelExecutor(infaas_query_grpc.QueryServicer):
  def __init__(self, args):
    self.model = None
    self.cuda = args.cuda  # If cuda=True, use GPU.
    self.model_name = None
    self.tokenizer = None
    self.beam_size = args.beam_size
    self.math = args.math

    self.device = utils.set_device(args.cuda, 0)  # no distributed, rank=0 always, 1 GPU
    utils.init_distributed(args.cuda)
    utils.setup_logging()  # no detailed logging.
    # Print debugging info.
    # utils.log_env_info()
    dtype = {'fp32': torch.FloatTensor, 'fp16': torch.HalfTensor}
    if not args.cuda and torch.cuda.is_available():
      warnings.warn('cuda is available but not enabled')

    try:
      # Load gnmt model
      model_path = os.path.join(model_directory, args.model)
      model_files = [fn for fn in os.listdir(model_path)
                    if fn.endswith(model_suffix)]

      start = now()
      self.model_name = args.model
      # First deserialize checkpoint to CPU (to save GPU memory)
      checkpoint = torch.load(os.path.join(model_path, model_files[0]),
                              map_location={'cuda:0': 'cpu'})

      # build GNMT model
      self.tokenizer = Tokenizer()
      self.tokenizer.set_state(checkpoint['tokenizer'])
      model_config = checkpoint['model_config']
      model_config['batch_first'] = True
      model_config['vocab_size'] = self.tokenizer.vocab_size
      self.model = GNMT(**model_config)
      self.model.load_state_dict(checkpoint['state_dict'])
      self.model.type(dtype[self.math])
      self.model = self.model.to(self.device)
      self.model.eval()

      # Build sequence generator
      self.generator = SequenceGenerator(
            model=self.model,
            beam_size=self.beam_size,
            max_seq_len=MAX_SEQ_LEN,
            len_norm_factor=0.6,
            len_norm_const=5.0,
            cov_penalty_factor=0.1)
      if self.beam_size == 1:
        self.generator = self.generator.greedy_search
      else:
        self.generator = self.generator.beam_search
      end = now()
      print('Loads model: {:.3f} ms'.format((end-start)*1000.0))
    except Exception as e:
      print(e)
      raise Exception(e)

    print('Finished loading: {}'.format(args.model))

  # Return a batch of Pytorch input tensor, and the lengths.
  def __load_inputs(self, inp):
    tensor_inp = None
    # Load input from raw text. Also pad the batch to the longest sentence.
    seq = []
    for raw in inp:
      tokenized = self.tokenizer.tokenize(raw)
      seq.append(tokenized)
      #print(tokenized)
    lengths = torch.tensor([len(s) for s in seq], dtype=torch.int64)
    batch_length = max(lengths)
    print('Maximum input length = {}'.format(batch_length))
    shape = (len(seq), batch_length)
    tensor_inp = torch.full(shape, config.PAD, dtype=torch.int64)
    for i, s in enumerate(seq):
      end_seq = lengths[i]
      tensor_inp[i, :end_seq].copy_(s[:end_seq])
    return (tensor_inp, lengths)

  def __make_prediction(self, inputs):
    output = []
    try:
      model_inp = self.__load_inputs(inputs)
      batch_size = len(inputs)
      #print(model_inp)
    except Exception as e:
      print(e)
      raise Exception(e)

    # Logic to do inference.
    insert_target_start = [config.BOS]
    bos = [insert_target_start] * (batch_size * self.beam_size)
    bos = torch.tensor(bos, dtype=torch.int64, device=self.device)
    # Batch first
    bos = bos.view(-1, 1)
    src, src_length = model_inp
    src = src.to(self.device)
    src_length = src_length.to(self.device)
    with torch.no_grad():
      context = self.model.encode(src, src_length)
      context = [context, src_length, None]
      preds, lengths, counter = self.generator(batch_size, bos, context)
    # Get the results on CPU.
    preds = preds.cpu()
    for pred in preds:
      pred = pred.tolist()
      dtok = self.tokenizer.detokenize(pred)
      #print(dtok)
      output.append(dtok)
    return output

  def QueryOnline(self, request, context):
    # Set nice value to 0.
    os.nice(0)
    print('raw_input length (batch) = {}'.format(len(request.raw_input)))

    startTime = now()
    output = self.__make_prediction(request.raw_input)
    reply = infaas_query.QueryOnlineResponse()
    reply.status.status = infaas_status.SUCCESS
    # Copy output to reply buffer
    for otp in output:
      print(otp)
      try:
        # Encode from str to bytes
        reply.raw_output.append(otp.encode())
      except Exception as e:
        print(e)
        raise Exception(e)
    stopTime = now()
    print('Infer took {:.4f} msec'.format((stopTime - startTime)*1000.0))

    return reply

  # Offline query
  def QueryOffline(self, request, context):
    print('Not implemented!')
    reply = infaas_query.QueryOfflineResponse()
    reply.status.status = infaas_status.INVALID
    reply.status.msg = 'Not implemented!'
    return reply

  # Heartbeat for health check of this container
  def Heartbeat(self, request, context):
    print('Received heartbeat request!')
    reply = infaas_query.HeartbeatResponse()
    reply.status.status = infaas_status.SUCCESS
    if (request.status.status != infaas_status.SUCCESS):
      print('Heartbeat request invalid status: {}'.format(request.status.status))
      reply.status.status = infaas_status.INVALID
      reply.status.msg = 'Invalid request status: {}'.format(request.status.status)

    return reply

def main(args):
  port = str(args.port)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
    options=[('grpc.max_send_message_length', MAX_GRPC_MSG_SIZE),
             ('grpc.max_receive_message_length', MAX_GRPC_MSG_SIZE)])
  infaas_query_grpc.add_QueryServicer_to_server(ModelExecutor(args), server)
  server.add_insecure_port('[::]:' + port)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, terminateHandler)
  signal.signal(signal.SIGTERM, terminateHandler)
  signal.signal(signal.SIGQUIT, terminateHandler)
  args = get_args()
  print(args)

  main(args)
