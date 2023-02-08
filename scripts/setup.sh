#!/bin/bash

# Script to install necessary dependencies for INFaaS
set -xe

sudo yum update
sudo yum install -y python-pip
pip install --user --upgrade pip
pip3 install --user --upgrade pip

sudo yum install -y  autoconf automake libtool

# install gcc and g++
sudo yum groupinstall -y "Development Tools"

# Install docker
./install_docker.sh

# Get the current version of the AWS CLI
current_version=$(aws --version | awk '{print $1}' | awk -F '/' '{print $2}')

# Check if the current version is not 2.x.x
if [[ ! $current_version =~ ^2 ]]; then
  echo "Current AWS CLI version is $current_version. Updating to the latest version..."
  pushd /tmp
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  sudo ./aws/install
  rm -rf awscliv2.zip
  popd
else
  echo "Current AWS CLI version is $current_version. Latest version already installed."
fi
