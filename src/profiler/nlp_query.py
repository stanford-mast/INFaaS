"""
Copyright 2018-2021 Board of Trustees of Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/python3
import numpy as np
import sys
import os
from timeit import default_timer as now

sys.path.insert(0, '../../protos/internal')
import grpc
import query_pb2 as infaas_query
import query_pb2_grpc as infaas_query_grpc

# Clip threshold; can be configured
clip_thresh = 20

def main():
  if len(sys.argv) != 5:
    print("Usage: ./nlp_query.py <input-sentences-path> <port> <batch-size> <is-measuring-loading>")
    sys.exit(1)

  input_sentence_path = sys.argv[1]
  port = sys.argv[2]
  batch_size = int(sys.argv[3])
  measure_load = int(sys.argv[4])

  # Load in all sentences and create a mapping as follows:
  # key: sentence length, value: [sentence 1, sentence 2, ...]
  sentence_map = {}
  fd = open(input_sentence_path, 'r')
  for line in fd:
    line_len = len(line.split())
    if line_len not in sentence_map:
      sentence_map[line_len] = []
    sentence_map[line_len].append(line.strip())
  fd.close()

  # Keep a running average of the performance
  avg_lat = 0
  for k,v in sentence_map.items():
    next_batch = []
    batch_iter = 0
    while batch_iter < batch_size:
      next_sentence = v[batch_iter % len(v)]
      input_str = str.encode(next_sentence)
      next_batch.append(input_str)
      batch_iter += 1

    start = now()
    with grpc.insecure_channel('localhost:' + port) as channel:
      stub = infaas_query_grpc.QueryStub(channel)
      request = infaas_query.QueryOnlineRequest(raw_input=next_batch)
      response = stub.QueryOnline(request)
    end = now()
    e2e = end - start
    avg_lat += e2e

    if measure_load or k > clip_thresh:
      break

  avg_lat = (avg_lat / len(sentence_map))

  print('%.4f' % avg_lat)

if __name__ == '__main__':
  main()
