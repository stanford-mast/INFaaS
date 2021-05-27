#!/usr/bin/python3
import os
import sys
import signal
import argparse
from concurrent import futures
import time
import struct
import numpy as np
import tensorflow as tf
import grpc
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import densenet
from timeit import default_timer as now

sys.path.insert(0, '../../protos/internal')
import infaas_request_status_pb2 as infaas_status
import query_pb2 as infaas_query
import query_pb2_grpc as infaas_query_grpc

model_directory = '/tmp/model'
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MAX_GRPC_MSG_SIZE = 2147483647

# tf.keras.backend.set_image_data_format('channels_last')

def terminateHandler(sigNum, frame):
  print('Received: {}, exit process.'.format(sigNum))
  sys.exit()

def get_args():
  ap = argparse.ArgumentParser()
  ap.add_argument('--scale', dest='scale', type=int, default=224,
                  help='Scale (e.g. 224) for image')
  ap.add_argument('--model', dest='model', default='testmodel',
                  help='Model name to load')
  ap.add_argument('--port', dest='port', type=int, default=9001,
                  help='Port to listen on')
  ap.add_argument('--cores', dest='cores', type=int, default=1,
                  help='Number of neuron cores to use. Must be within [1, 4]. Default=1')

  return ap.parse_args()

class ModelExecutor(infaas_query_grpc.QueryServicer):
  def __init__(self, scale, ser_model):
    self.model = None
    try:
      # Load Neuron-tf model
      model_path = os.path.join(model_directory, ser_model)
      start = now()
      self.model_name = ser_model
      self.model = tf.contrib.predictor.from_saved_model(model_path)
      end = now()
      print('Loads model: {:.3f} ms'.format((end-start)*1000.0))
    except Exception as e:
      print(e)
      raise Exception(e)

    self.scale = scale
    print('Finished loading: {}'.format(ser_model))

  def __load_inputs(self, inp):
    tensor_inp = None
    for i in inp:
      # TODO: Check NCHW, or NHWC
      tmparr = np.asarray(struct.unpack('%sf' % (len(i)//4), i), dtype=np.float32).reshape(3, self.scale, self.scale)
      if tensor_inp is None:
        tensor_inp = np.expand_dims(tmparr, axis=0)
      else:
        tensor_inp = np.vstack((tensor_inp, tmparr))
    # TODO: check models use NCHW or NHWC. The following is to change from NCHW to NHWC.
    tensor_inp = np.transpose(tensor_inp, (0, 2, 3, 1))
    # TODO: maybe we can do preprocess on the client side. Now I just input
    # unscaled raw image.
    if 'v2' in self.model_name or 'v3' in self.model_name:
      tensor_inp = resnet_v2.preprocess_input(tensor_inp)
    elif 'dense' in self.model_name:
      tensor_inp = densenet.preprocess_input(tensor_inp)
    else:
      tensor_inp = resnet50.preprocess_input(tensor_inp)
    print(tensor_inp.shape)
    return tensor_inp

  def __make_prediction(self, inputs):
    try:
      tensor_inp = self.__load_inputs(inputs)
    except Exception as e:
      print(e)
      raise Exception(e)
    model_feed_dict = {'input': tensor_inp}
    output = self.model(model_feed_dict)
    if 'v2' in self.model_name or 'v3' in self.model_name:
      print(resnet_v2.decode_predictions(output["output"], top=5)[0])
    elif 'dense' in self.model_name:
      print(densenet.decode_predictions(output["output"], top=5)[0])
    else:
      print(resnet50.decode_predictions(output["output"], top=5)[0])
    return output["output"]

  def QueryOnline(self, request, context):
    # Set nice value to 0.
    # TODO: implement the TF serving logic
    os.nice(0)
    print('raw_input length (batch) = {}'.format(len(request.raw_input)))
    print('raw input size = {}'.format(len(request.raw_input[0])))

    startTime = now()
    output = self.__make_prediction(request.raw_input)
    reply = infaas_query.QueryOnlineResponse()
    reply.status.status = infaas_status.SUCCESS
    print(type(output))
    # Copy output to reply buffer
    for otp in output:
      print(otp.shape)
      # TODO: may nee to convert format for real output data.
      raw_list = np.asarray(otp, dtype=np.float32).flatten()
      print('output size = {}'.format(len(raw_list)))
      raw_list = struct.pack('%sf' % len(raw_list), *raw_list)
      print(len(raw_list))

      try:
        reply.raw_output.append(raw_list)
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
  # TODO: may need more than one worker?
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
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
  signal.signal(signal.SIGINT, terminateHandler)
  signal.signal(signal.SIGTERM, terminateHandler)
  signal.signal(signal.SIGQUIT, terminateHandler)
  args = get_args()
  assert args.cores >= 1
  assert args.cores <= 4
  print(args)
  # Specify number of neuron cores to use. Default 1.
  os.environ['NEURONCORE_GROUP_SIZES'] = str(args.cores)
  print("NEURONCORE_GROUP_SIZES (env): " + os.environ.get('NEURONCORE_GROUP_SIZES', '<unset>'))

  main(args)
