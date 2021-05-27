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
import matplotlib.image as mpimg
import numpy as np
import sys
import os
from timeit import default_timer as now

sys.path.insert(0, '../../protos/internal')
import grpc
import query_pb2 as infaas_query
import query_pb2_grpc as infaas_query_grpc

def main():
  if len(sys.argv) != 4:
    print("Usage: ./pytorch_query.py <image-path> <port> <batch-size>")
    sys.exit(1)

  image_path = sys.argv[1]
  port = sys.argv[2]
  batch_size = int(sys.argv[3])
  if not os.path.exists(image_path):
    print(image_path, "is not a valid path")
    sys.exit(1)

  img = mpimg.imread(image_path)
  img_flatten = img.flatten().astype(np.float32)
  img_bytes = img_flatten.tobytes()

  input_arr = [img_bytes] * batch_size

  start = now()
  with grpc.insecure_channel('localhost:' + port) as channel:
    stub = infaas_query_grpc.QueryStub(channel)
    request = infaas_query.QueryOnlineRequest(raw_input=input_arr)
    response = stub.QueryOnline(request)
  end = now()
  e2e = end - start
  #e2e_ms = e2e * 1000
  print('%.4f' % e2e)

if __name__ == '__main__':
  main()
