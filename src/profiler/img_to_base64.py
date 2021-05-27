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
import base64
import sys
import os

def main():
  if len(sys.argv) != 2:
    print("Usage: ./img_to_base64.py <image-path>")
    sys.exit(1)

  image_path = sys.argv[1]
  if not os.path.exists(image_path):
    print(image_path, "is not a valid path")
    sys.exit(1)

  img = mpimg.imread(image_path)
  img_flatten = img.flatten().astype(np.float32)
  img_bytes = img_flatten.tobytes()
  b64_enc = base64.b64encode(img_bytes)
  b64_string = str(b64_enc)

  # Print for caller to grab
  print(b64_string)

if __name__ == '__main__':
  main()
