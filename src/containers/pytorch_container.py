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

# PyTorch Container

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import tempfile
import argparse
import struct
import torch
import sys
import os
from concurrent import futures
import subprocess as sp
import grpc
from timeit import default_timer as now
import time

sys.path.insert(0, '../../protos/internal')
import infaas_request_status_pb2 as infaas_status
import query_pb2 as infaas_query
import query_pb2_grpc as infaas_query_grpc

model_directory = '/tmp/model'
input_root_dir = '/tmp/infaas_input'
output_root_dir = '/tmp/infaas_output'
input_leaf_dir = 'infer'
model_suffix = 'pt'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_GRPC_MSG_SIZE = 2147483647

def get_args():
  ap = argparse.ArgumentParser()
  ap.add_argument('scale', type=int,
                  help='Scale (e.g. 224) for image')
  ap.add_argument('model',
                  help='Serialized model to load')
  ap.add_argument('port', type=int,
                  help='Port to listen on')

  return ap.parse_args()

class ModelExecutor(infaas_query_grpc.QueryServicer):
  def __init__(self, scale, ser_model):
    self.model = None
    try:
      # NOTE: the ser_model should be a directory, and we need to find the
      # model file *.pt inside that directory.
      model_path = os.path.join(model_directory, ser_model)
      model_files = [fn for fn in os.listdir(model_path)
                    if fn.endswith(model_suffix)]
      # There should only be one file.
      assert len(model_files) == 1
      self.model = torch.load(os.path.join(model_path, model_files[0]))
      self.model.eval()
    except Exception as e:
      print(e)
      raise StandardError

    self.scale = scale
    print('Finished loading: {}'.format(ser_model))

  def __load_inputs(self, inp):
    # Using // instead of / is important for using integer division instead of floating point
    tensor_inp = torch.stack([torch.Tensor(np.asarray(struct.unpack('%sf' % (len(i)//4), i), dtype=np.float32)).reshape(3,self.scale,self.scale) for i in inp])
    return tensor_inp

  # Load input files from inp_dir with batch size.
  def __load_directory(self, inp_dir, batch):
    val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(inp_dir,
      transforms.Compose([transforms.CenterCrop(self.scale),
                          transforms.ToTensor()])),
      batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    return val_loader

  # inputs are raw_input from the Online request.
  def __make_prediction(self, inputs):
    print('inputs length (batch) = {}'.format(len(inputs)))
    tensor_inp = self.__load_inputs(inputs)
    with torch.no_grad():
      input = torch.autograd.Variable(tensor_inp)
      output = self.model(input)
      print('output batch is - {}'.format(len(output)))
      return output

  # Loader is a torch DataLoader
  def __infer_from_loader(self, loader):
    all_out = []
    with torch.no_grad():
      for (input, _) in loader:
        input = torch.autograd.Variable(input)
        output = self.model(input)
        print('output batch is - {}'.format(len(output)))
        all_out.append(output)
    return all_out

  # For Online path
  def QueryOnline(self, request, context):
    # Set nice value to 0.
    os.nice(0)
    print('raw_input length (batch) = {}'.format(len(request.raw_input)))
    print('raw input size = {}'.format(len(request.raw_input[0])))
    startTime = now()
    output = self.__make_prediction(request.raw_input)
    reply = infaas_query.QueryOnlineResponse()
    for otp in output:
      raw_list = np.asarray(otp, dtype=np.float32).flatten()
      print('output size = {}'.format(len(raw_list)))
      raw_list = struct.pack('%sf' % len(raw_list), *raw_list)
      print(len(raw_list))
      try:
        reply.raw_output.append(raw_list)
      except Exception as e:
        print(e)
        raise StandardError
    stopTime = now()
    print('Infer took {:.4f} msec'.format((stopTime - startTime)*1000.0))
    print('reply raw output size = {}'.format(len(reply.raw_output)))
    return reply

  # For Offline path, the input_url will be input directory (local); and the
  # output_url will be output directory (local).
  def QueryOffline(self, request, context):
    offline_nice = os.getenv('OFFLINE_NICE', 'OFF')
    if offline_nice == 'ON':
      # Set nice value to 10, to lower priority.
      os.nice(10)
    else:
      os.nice(0)
    print('OFFLINE_NICE: {}'.format(offline_nice))
    inp_dir = os.path.join(input_root_dir, request.input_url)
    out_dir = os.path.join(output_root_dir, request.output_url)

    # NOTE: Need one more level for the DataLoader: root_dir/inp_dir/infer/xxx.png
    real_input_path = os.path.join(inp_dir, input_leaf_dir)
    reply = infaas_query.QueryOfflineResponse()

    # Make sure input and output directories are correct.
    if ((not inp_dir) or (not os.path.isdir(real_input_path))):
      print('Invalid input directory provided!')
      reply.status.status = infaas_status.INVALID
      reply.status.msg = 'Invalid input directory: {}'.format(inp_dir)
      return reply
    if ((not out_dir) or (not os.path.isdir(out_dir))):
      print('Invalid output directory provided!')
      reply.status.status = infaas_status.INVALID
      reply.status.msg = 'Invalid output directory: {}'.format(out_dir)
      return reply

    print('Serving offline request.\n')
    print('Input directory: {}'.format(inp_dir))
    print('Output directory: {}'.format(out_dir))

    startTime = now()
    inp_names = [img for img in os.listdir(real_input_path) if os.path.isfile(os.path.join(real_input_path, img))]
    print('Loading total {} input'.format(len(inp_names)))
    # For now, process all files in one batch. We assume the offline
    # worker can control how many files to write for each iteration.
    loader = self.__load_directory(inp_dir, len(inp_names))
    all_out = self.__infer_from_loader(loader)

    # Postprocess output
    ind = 0
    for one_iter in all_out:
      for otp in one_iter:
        raw_list = np.asarray(otp, dtype=np.float32).flatten()
        file_name = inp_names[ind]
        print('output size for {} = {}'.format(file_name, len(raw_list)))
        raw_list = struct.pack('%sf' % len(raw_list), *raw_list)
        file_path = os.path.join(out_dir, file_name + '.out')
        try:
          with open(file_path, "wb") as fh:
            fh.write(raw_list)
        except Exception as e:
          print(e)
          raise StandardError
        ind += 1
    reply.status.status = infaas_status.SUCCESS
    stopTime = now()
    print('Infer took {:.4f} msec'.format((stopTime - startTime)*1000.0))

    return reply

  # Heartbeat for health check of this container
  def Heartbeat(self, request, context):
    reply = infaas_query.HeartbeatResponse()
    reply.status.status = infaas_status.SUCCESS
    if (request.status.status != infaas_status.SUCCESS):
      print('Heartbeat request invalid status: {}'.format(request.status.status))
      reply.status.status = infaas_status.INVALID
      reply.status.msg = 'Invalid request status: {}'.format(request.status.status)

    return reply

def main(args):
  # server = msgpackrpc.Server(ModelPredictor(args.scale, args.model))
  # server.listen(msgpackrpc.Address("localhost", args.port))
  # server.start()
  port = str(args.port)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
    options=[('grpc.max_send_message_length', MAX_GRPC_MSG_SIZE),
             ('grpc.max_receive_message_length', MAX_GRPC_MSG_SIZE)])
  infaas_query_grpc.add_QueryServicer_to_server(ModelExecutor(args.scale, args.model), server)
  server.add_insecure_port('[::]:' + port)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  main(get_args())

